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
            {'name': 'black_border', 'type': 'single', 'file': 'black_border.npy'}, ## For not showing sprites at ends (x < 3, x > 207) of the screen.
            {'name': 'ground', 'type': 'single', 'file': 'ground.npy'}, ## TODO: Ground and Background the same sprite?
            {
                'name': 'car', 'type': 'group',
                ## All three recolors feed the death color-cycle
                'files': ['car.npy', 'car_dead_1.npy', 'car_dead_2.npy', 'car_dead_3.npy']
            },
            {'name': 'satellite', 'type': 'single', 'file': 'satellite.npy'},
            {
                'name': 'helicopter', 'type': 'group',
                'files': ['helicopter_1.npy', 'helicopter_2.npy']
            },
            {
                'name': 'helicopter_melee', 'type': 'group',
                ## The searchlight sweep table indexes sprites 0..14, so the
                ## whole extracted sequence has to be here (the group loader
                ## pads the differing widths).
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
            ## Water scene terrain and actors
            {'name': 'water', 'type': 'single', 'file': 'water.npy'},
            ## seabed_full is the complete 160px repeating strip including
            ## the 14-column valley gap; the old seabed.npy had the gap
            ## columns deleted, which tiled a valley-less seabed.
            {'name': 'seabed', 'type': 'single', 'file': 'seabed_full.npy'},
            {'name': 'water_sky', 'type': 'single', 'file': 'water_sky.npy'}, ## solid 74,74,74 measured in ALE
            ## The splash frogman's two poses, both pixel-exact extractions
            {'name': 'splash', 'type': 'single', 'file': 'explosion_1_(small).npy'}, ## TODO: Keep radiation sprites together
            {'name': 'splash_wide', 'type': 'single', 'file': 'explosion_2.npy'},
            ## A bolt sinking past a living frogman is drawn in his colors
            {'name': 'laser_green', 'type': 'single', 'file': 'laser_green.npy'},
            ## Second water scene: darker water and its roster, all cropped
            ## from real ALE frames of that scene
            {'name': 'water_b', 'type': 'single', 'file': 'water_b.npy'},
            {'name': 'sky_flash', 'type': 'single', 'file': 'sky_flash.npy'}, ## whole sky flashes gray when the rocket bursts
            {'name': 'death_flash', 'type': 'single', 'file': 'death_flash.npy'}, ## the sky's one-frame flash on a water death
            {'name': 'rocket', 'type': 'single', 'file': 'rocket.npy'},
            {'name': 'submarine', 'type': 'single', 'file': 'submarine.npy'},
            ## Water B's pink ball (the old "pink helicopter"): a 9x11
            ## sphere block-sampled from the longplay, solid and striped
            ## poses alternating in flight; the striped one is also its
            ## hit pose during the scene-clear freeze.
            {
                'name': 'wb_ball', 'type': 'group',
                'files': ['wb_ball_1.npy', 'wb_ball_2.npy']
            },
            {'name': 'flyer_red', 'type': 'single', 'file': 'flyer_red.npy'},
            ## Third water scene ("water C"): daylight sky, navy water and a
            ## green seabed (solids and the recolored seabed strip come from
            ## the longplay video), plus the roster only this scene has.
            {'name': 'wc_sky', 'type': 'single', 'file': 'wc_sky.npy'},
            {'name': 'wc_water', 'type': 'single', 'file': 'wc_water.npy'},
            {'name': 'wc_seabed', 'type': 'single', 'file': 'wc_seabed.npy'},
            {'name': 'wc_cloud', 'type': 'single', 'file': 'wc_cloud.npy'},
            {'name': 'wc_ship', 'type': 'single', 'file': 'wc_ship.npy'},
            ## The daylight rocket is a small 5x8 gray pyramid (block-sampled
            ## from the video); the flame version shows while it climbs
            {'name': 'wc_rocket', 'type': 'single', 'file': 'wc_rocket.npy'},
            {'name': 'wc_rocket_fire', 'type': 'single', 'file': 'wc_rocket_fire.npy'},
            {'name': 'wc_base', 'type': 'single', 'file': 'wc_base.npy'},
            {'name': 'wc_goal', 'type': 'single', 'file': 'wc_goal.npy'},
            {
                'name': 'scuba', 'type': 'group',
                'files': ['scuba_1.npy', 'scuba_2.npy']
            },

            {'name': 'life', 'type': 'single', 'file': 'car_life.npy'},
            {'name': 'oil_rig', 'type': 'group', 'files': ['oil_rig.npy', 'oil_rig.npy']},

            {
                'name': 'score_digits', 'type': 'digits',
                'pattern': 'score_{}.npy'
            },
            {
                'name': 'bullet', 'type': 'group',
                'files': ['bullet.npy', 'w_bullet.npy'] ## w_bullet is the player's water bullet
            },
            ## The submarine's shot: two yellow 2x2 dots stacked with a gap
            ## (block-sampled from the longplay)
            {'name': 'sub_shot', 'type': 'single', 'file': 'sub_shot.npy'},
            ## Rocket debris that reached the daylight waterline: a red /
            ## pink sparkle alternating between two dot patterns (video)
            {
                'name': 'wc_splash', 'type': 'group',
                'files': ['wc_splash_1.npy', 'wc_splash_2.npy']
            },
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
    GAME_AREA_MIN_X: int = struct.field(pytree_node=False, default=4) ## Playable Area: 5 (Coordinate system starting with 1)
    GAME_AREA_MAX_X: int = struct.field(pytree_node=False, default=73) ## Playable Area: 81 (Coordinate system starting with 1)
    GAME_AREA_MIN_Y: int = struct.field(pytree_node=False, default=0) ## Playable Area: 123 (Top-left coordinate system); 87 (Bottom-right co-sys)
    GAME_AREA_MAX_Y: int = struct.field(pytree_node=False, default=119)
    ## GAME_AREA_MAX_X above is the PLAYER's hard stop (measured: the hull
    ## stops exactly at columns 73-80). World objects use the real screen
    ## edges: they enter on the right around column 150-158 and leave on
    ## the left, exactly like in ALE.
    OBJECT_SPAWN_X: int = struct.field(pytree_node=False, default=150) ## helicopter entry column
    OBJECT_SPAWN_X_FAR: int = struct.field(pytree_node=False, default=158) ## diamond / pit entry column
    OBJECT_EXIT_X: int = struct.field(pytree_node=False, default=159) ## rightward movers leave here

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

    MAX_LIVES: int = struct.field(pytree_node=False, default=6) ## the real game starts with 6 (both ALE agents measured it)
    MAX_DIAMONDS: int = struct.field(pytree_node=False, default=8)
    MAX_ENEMIES: int = struct.field(pytree_node=False, default=8)
    MAX_HELICOPTERS: int = struct.field(pytree_node=False, default=4)
    MAX_SATELLITES: int = struct.field(pytree_node=False, default=4)
    ## Reaching the second water scene takes ~9000 clean frames (land
    ## 4454 + water 4435 + death freezes), so the old 5000 cap ended every
    ## episode before the dock bonus could ever pay out.
    ## Water B (~4435 more) and the daylight scene (~4000 + the ending)
    ## push a full run past 17k clean frames, so leave room for deaths.
    MAX_EPISODE_STEPS: int = struct.field(pytree_node=False, default=30000)

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
    ## Fire pit sprite size, re-extracted from real frames: the crater top
    ## pokes 2px above the road and the ember tail reaches ~38 rows down.
    PIT_WIDTH: int = struct.field(pytree_node=False, default=16)
    PIT_HEIGHT: int = struct.field(pytree_node=False, default=40)

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
    ## Losing a life freezes the whole scene for exactly this many frames
    ## (measured: sprite color-cycles, scroll and every actor stand still),
    ## then the player respawns and the sky actors are cleared.
    DEATH_ANIMATION_FRAMES: int = struct.field(pytree_node=False, default=59)

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
    HELICOPTER_BOMB_RANGE: int = struct.field(pytree_node=False, default=75) ## drops observed at 46-75px on the approach side
    HELICOPTER_BOMB_RETRY_FRAMES: int = struct.field(pytree_node=False, default=33) ## same-pass re-drops measured 29-37 frames apart
    HELICOPTER_BOMB_MAX_PER_PASS: int = struct.field(pytree_node=False, default=4)
    ## The real picker looks like the ROM's internal random generator (it's
    ## not position, speed or the missile slot, we tested all three). So:
    ## while the heli is in range it gets a chance every RETRY_FRAMES, each
    ## one rarer than the last (chance / (1 + drops so far)), which lands at
    ## roughly: one bomb common, two rarer, three much rarer, four rare.
    HELICOPTER_BOMB_DROP_CHANCE: float = struct.field(pytree_node=False, default=0.5)
    ## Land cadence: the pass is ~204 frames and the real game usually
    ## drops 2 lasers per pass at 30-90 frame spacings; a 75-frame timer
    ## lands on 2 drops per pass (52 was giving 3-4).
    SATELLITE_LASER_DROP_PERIOD: int = struct.field(pytree_node=False, default=75)
    SATELLITE_LASER_FALL_SPEED: int = struct.field(pytree_node=False, default=1) ## laser falls straight down, no sideways drift

    ## Stage progression, measured clean (death freezes subtracted) in ALE:
    ## the land scene runs ~4454 frames before the terrain turns to water;
    ## the first water scene runs ~4435 frames and ends at a dock that pays
    ## a 5000 point bonus; after it comes the second water scene (darker
    ## water, new enemy set) which never ended in a 22000 frame probe, so
    ## it is treated as endless here.
    ## START_STAGE jumps a fresh game straight into a later scene for
    ## playtesting (0 land, 1 first water, 2 second water, 3 daylight
    ## water). Settable without code changes via the JB_START_STAGE
    ## environment variable:
    ##   JB_START_STAGE=3 python scripts/play.py -g jamesbond
    START_STAGE: int = struct.field(pytree_node=False, default=0)
    STAGE_ONE_LENGTH: int = struct.field(pytree_node=False, default=4454)
    STAGE_TWO_LENGTH: int = struct.field(pytree_node=False, default=4435)
    ## Scene names used in comments and docs (stage index in brackets):
    ##   [0] land, [1] water A with the oil rig, [2] water B, the dark
    ##   water with the big rockets and the pink balls, [3] water C, the
    ##   daylight scene. [4] is only the terminal marker after the last
    ##   bonus. Every scene ends with the same 59-frame freeze + colour
    ##   cycle and a 5000 bonus: landing on the rig, shooting enough pink
    ##   balls, diving into the daylight objective.
    SCORE_STAGE_BONUS: int = struct.field(pytree_node=False, default=5000)

    ## Water scene: scuba diver. Measured in ALE frame by frame (two
    ## agents cross-checked): a vertical swimmer, 7 wide x 20 tall
    ## (scuba_1/2.npy are pixel exact), who enters from the right screen
    ## edge at a fixed depth below the surface, swims left 1px every 4th
    ## frame, never chases, alternates his two animation frames every 15
    ## frames, and vanishes mid-screen on an age clock rather than at the
    ## left edge. He cannot be shot (the player's only round is the
    ## up-forward anti-air shot); running into him under water costs a
    ## life. A boat on the surface floats above his body.
    SCUBA_WIDTH: int = struct.field(pytree_node=False, default=7)
    SCUBA_HEIGHT: int = struct.field(pytree_node=False, default=20)
    SCUBA_SPAWN_X: int = struct.field(pytree_node=False, default=155) ## right screen edge
    SCUBA_SPAWN_Y: int = struct.field(pytree_node=False, default=129) ## body below the surface row (ALE 131, our rows sit 2 higher)
    SCORE_SCUBA: int = struct.field(pytree_node=False, default=200)
    ## The deep swimmer's own clock (both clean tracked episodes of the
    ## 7x20 vertical diver ran 333 frames); the ~120 frame figure floating
    ## around belongs to the surface splash creature, not to him.
    SCUBA_LIFETIME_FRAMES: int = struct.field(pytree_node=False, default=333)
    SCUBA_RESPAWN_FRAMES: int = struct.field(pytree_node=False, default=150) ## breather between divers
    ## Radioactivity in the first water scene has two independent rules:
    ##   no diver on screen             -> a spent satellite bolt may create
    ##                                     the radioactive surface splash
    ##   diver anywhere on screen       -> the bolt disappears normally
    ##   diver near the player's boat   -> the DIVER becomes radioactive
    ## Proximity is checked every frame; it does not wait for a satellite
    ## bolt to land. Distance is horizontal because the diver remains below
    ## the surface while the boat can jump or dive.
    SCUBA_RADIOACTIVE_RANGE: int = struct.field(pytree_node=False, default=40)
    SCUBA_RADIOACTIVE_FRAMES: int = struct.field(pytree_node=False, default=120) ## how long he stays radioactive
    ## The laser bolt splashes THROUGH the surface: it keeps falling under
    ## water and detonates into a static green surface explosion that
    ## rides the world scroll, blocks the lane for a while, and kills the
    ## boat on near-contact (measured: adjacency within ~1px kills, even
    ## submerged; only a clearly airborne boat passes safely).
    SPLASH_WIDTH: int = struct.field(pytree_node=False, default=20)  ## explosion_1_(small).npy is 20x7
    SPLASH_HEIGHT: int = struct.field(pytree_node=False, default=7)
    SPLASH_Y: int = struct.field(pytree_node=False, default=123) ## straddles the surface row
    SPLASH_LIFETIME_FRAMES: int = struct.field(pytree_node=False, default=120)
    SPLASH_SAFE_PLAYER_Y: int = struct.field(pytree_node=False, default=119) ## airborne above this is safe
    WATER_LASER_FLOOR: int = struct.field(pytree_node=False, default=132) ## bolt sinks this deep before detonating
    ## Oil rig, sprite is a static 16x22
    OIL_RIG_WIDTH: int = struct.field(pytree_node=False, default=16)
    OIL_RIG_HEIGHT: int = struct.field(pytree_node=False, default=22)
    OIL_RIG_APPEAR_FRAME: int = struct.field(pytree_node=False, default=420) ## frames into the water scene before the rig shows
    OIL_RIG_SEQ_TOTAL: int = struct.field(pytree_node=False, default=150)     ## total frames of the appear-right/gap/appear-left sequence
    OIL_RIG_SEQ_RIGHT_END: int = struct.field(pytree_node=False, default=120)  ## seq value where the RIGHT phase ends
    OIL_RIG_SEQ_LEFT_START: int = struct.field(pytree_node=False, default=105) ## seq value where the LEFT phase begins
    OIL_RIG_RIGHT_X: int = struct.field(pytree_node=False, default=120)       ## right appear column
    OIL_RIG_LEFT_X: int = struct.field(pytree_node=False, default=45)         ## left appear column (player is to its right)
    OIL_RIG_Y: int = struct.field(pytree_node=False, default=100)             ## top-left y: deck at waterline, legs in water
    OIL_RIG_STRIKE_LEN: int = struct.field(pytree_node=False, default=24)     ## flash length; also the rig visible window
    OIL_RIG_MIN_STAGE1_STEPS: int = struct.field(pytree_node=False, default=1500) ## rig can't appear until 1500+ steps into the water scene
    OIL_RIG_RIGHT_SLIDE: int = struct.field(pytree_node=False, default=12)    ## px the rig drifts left while visible on the right
    OIL_RIG_TOP_LAND_MARGIN: int = struct.field(pytree_node=False, default=4) ## how close to the rig top counts as landing
    OIL_RIG_FLASH_FRAMES: int = struct.field(pytree_node=False, default=60) ## how long the rig is glimpsed in the flash
    OIL_RIG_STRIKE_FRAMES: int = struct.field(pytree_node=False, default=75) ## brief bright flash length

    ## Second water scene (after the dock bonus): darker water and a fresh
    ## enemy roster, all sprites cropped from real ALE frames. The floating
    ## rocket is the only scoring object (+200 for ramming it, though the
    ## ram usually costs a life too); after idling on the surface a while
    ## it ignites and launches skyward. The submarine cruises underwater
    ## and only threatens a diving boat; the two flyers cross the sky.
    SCORE_ROCKET: int = struct.field(pytree_node=False, default=200)
    ## The rocket's full launch cycle was measured frame by frame: it
    ## appears mid-screen already submerged (tip row ~140), floats a few
    ## frames, climbs at exactly 1px/frame straight up (a single-frame
    ## splash blip marks the waterline crossing), and EXPLODES with its
    ## tip at row 61 into two red debris bars that linger in the sky,
    ## plus a 1-2 frame full-sky gray flash. One rocket every 256 frames.
    ROCKET_WIDTH: int = struct.field(pytree_node=False, default=8)
    ROCKET_HEIGHT: int = struct.field(pytree_node=False, default=11)
    ROCKET_Y: int = struct.field(pytree_node=False, default=138) ## rests low in the water; a diving boat can ram it
    ROCKET_SPAWN_X: int = struct.field(pytree_node=False, default=85) ## appears mid-screen, not at the edge
    ROCKET_IGNITE_AGE: int = struct.field(pytree_node=False, default=6) ## floats briefly, then climbs
    ROCKET_EXPLODE_Y: int = struct.field(pytree_node=False, default=61) ## tip row where it bursts
    ROCKET_RESPAWN_FRAMES: int = struct.field(pytree_node=False, default=171) ## 256 frame cycle minus ~85 frames of life
    DEBRIS_LIFETIME_FRAMES: int = struct.field(pytree_node=False, default=120) ## ALE-measured linger; unused since the bars now fall like the daylight ones
    SKY_FLASH_FRAMES: int = struct.field(pytree_node=False, default=2) ## whole sky flashes gray on the burst
    SUBMARINE_WIDTH: int = struct.field(pytree_node=False, default=16)
    SUBMARINE_HEIGHT: int = struct.field(pytree_node=False, default=11)
    SUBMARINE_Y: int = struct.field(pytree_node=False, default=135) ## deep under the surface
    SUBMARINE_RESPAWN_FRAMES: int = struct.field(pytree_node=False, default=260)
    ## The pink ball (fields still called wb_heli_*): enters at the LEFT
    ## edge at row 57 and crosses to the right at 1.75 px/f (7 px every 4
    ## frames, read off the longplay at 60 fps). The anti-air shot pops
    ## it for 500; the shot that reaches WB_BALL_HITS_TO_EXIT ends the
    ## scene: the ball hangs in its striped pose while the scene freezes
    ## for the usual 59 frames, the sky flickers dark/light for the first
    ## ~21 of them, then the 5000 bonus pays and the daylight scene
    ## starts. The footage shows three 500-point hits before the exit
    ## (6000->6500, 6700->7200, 7600->8100); the team's count was two,
    ## so this is one constant to flip.
    WB_HELI_Y: int = struct.field(pytree_node=False, default=57)
    WB_HELI_WIDTH: int = struct.field(pytree_node=False, default=9)
    WB_HELI_HEIGHT: int = struct.field(pytree_node=False, default=11)
    WB_HELI_RESPAWN_FRAMES: int = struct.field(pytree_node=False, default=200)
    SCORE_BALL: int = struct.field(pytree_node=False, default=500)
    WB_BALL_HITS_TO_EXIT: int = struct.field(pytree_node=False, default=3)
    WB_EXIT_FLICKER_FRAMES: int = struct.field(pytree_node=False, default=21)
    ## The "small red flyer" of the early survey turned out to be the
    ## rocket's explosion debris (two stacked 4x2 red bars), not an enemy
    ## of its own: it spawns where the rocket bursts and drifts harmlessly.
    WB_FLYER_Y: int = struct.field(pytree_node=False, default=61)
    WB_FLYER_WIDTH: int = struct.field(pytree_node=False, default=4)
    WB_FLYER_HEIGHT: int = struct.field(pytree_node=False, default=5)
    ## Submarine, water B and C alike (longplay, 60 fps): it enters from
    ## the LEFT edge and cruises right, 2 px every 3 frames. Once per pass,
    ## as it crosses SUB_FIRE_X, it fires a double-dot shot from its bow
    ## that runs diagonally back and up ((-2,-1) px/frame, the "back-
    ## upward" leg) until it reaches SUB_SHOT_LEVEL_Y just under the
    ## surface, then straight left along that row at 4 px every 3 frames
    ## until it leaves the screen. It costs a life on contact: it passes
    ## under a surfaced hull but crosses a diving boat's path.
    SUB_SPAWN_X: int = struct.field(pytree_node=False, default=-12)
    SUB_FIRE_X: int = struct.field(pytree_node=False, default=70)
    SUB_SHOT_LEVEL_Y: int = struct.field(pytree_node=False, default=124)
    SUB_SHOT_WIDTH: int = struct.field(pytree_node=False, default=2)
    SUB_SHOT_HEIGHT: int = struct.field(pytree_node=False, default=5)

    ## Third water scene ("water C", the daylight one after water B). All
    ## numbers below were read frame by frame off a longplay recording
    ## (rows calibrated on the waterline, good to ~3 rows; speeds from
    ## 5-60 fps samples), NOT from ALE -- refine them when the ROM's exit
    ## from water B is found and the scene can be probed directly.
    WC_LENGTH: int = struct.field(pytree_node=False, default=3960)       ## frames until the sunken base scrolls in
    ## The scene runs in two phases: first rockets and the submarine, then
    ## the steamship takes the rockets' place (the submarine and the
    ## helicopter stay). The video switches somewhere in a scrubbed gap
    ## 44-53 s in, so this is a rough middle value.
    WC_SHIP_PHASE_START: int = struct.field(pytree_node=False, default=2800)
    ## Rockets: up to two launch pads sit in the water at once, at random
    ## columns, each riding the seabed scroll until its own launch moment.
    ## The daylight pyramid is smaller than water B's: 5 wide, 8 tall.
    WC_ROCKET_WIDTH: int = struct.field(pytree_node=False, default=5)
    WC_ROCKET_HEIGHT: int = struct.field(pytree_node=False, default=8)
    ## They are the scene's only regular score: either player round
    ## destroys one. An unshot rocket bursts at the usual row; its red
    ## debris then FALLS back to the waterline (1px/frame), floats there
    ## for a moment and vanishes -- and it is lethal on the way down.
    WC_ROCKET_SLOTS: int = struct.field(pytree_node=False, default=2)
    WC_ROCKET_SPAWN_MIN_X: int = struct.field(pytree_node=False, default=20)
    WC_ROCKET_SPAWN_MAX_X: int = struct.field(pytree_node=False, default=140)
    WC_ROCKET_IDLE_MIN: int = struct.field(pytree_node=False, default=60)  ## frames a pad rides the seabed before launching
    WC_ROCKET_IDLE_MAX: int = struct.field(pytree_node=False, default=180)
    WC_ROCKET_GAP_MIN: int = struct.field(pytree_node=False, default=60)   ## breather before a slot refills
    WC_ROCKET_GAP_MAX: int = struct.field(pytree_node=False, default=200)
    SCORE_ROCKET_SHOT: int = struct.field(pytree_node=False, default=100)
    WC_DEBRIS_REST_Y: int = struct.field(pytree_node=False, default=119)
    WC_DEBRIS_REST_FRAMES: int = struct.field(pytree_node=False, default=40)
    ## The depth charge sinks the submarine (seen twice in the daylight
    ## footage, +200 each time)
    SCORE_SUBMARINE_SHOT: int = struct.field(pytree_node=False, default=200)
    ## The daylight anti-air shot also pops the falling debris for +100
    ## (video: 13500 -> 13600 right after a burst)
    SCORE_DEBRIS_SHOT: int = struct.field(pytree_node=False, default=100)
    WC_SPLASH_WIDTH: int = struct.field(pytree_node=False, default=6)     ## the waterline sparkle's footprint
    WC_SPLASH_HEIGHT: int = struct.field(pytree_node=False, default=3)
    WC_SPLASH_Y: int = struct.field(pytree_node=False, default=118)
    WC_SPLASH_FLIP_FRAMES: int = struct.field(pytree_node=False, default=6) ## pattern swap cadence
    ## Steamship riding the waterline right to left at the helicopter's
    ## 0.6px/f, one pass right after the other. Only its deck and hull
    ## (the lower rows) are solid, so a jump clears it and a dive ducks it.
    WC_SHIP_WIDTH: int = struct.field(pytree_node=False, default=16)
    WC_SHIP_HEIGHT: int = struct.field(pytree_node=False, default=16)
    WC_SHIP_Y: int = struct.field(pytree_node=False, default=106)        ## hull rows 118-121 sit on the waterline
    WC_SHIP_HULL_TOP: int = struct.field(pytree_node=False, default=114)
    WC_SHIP_RESPAWN_FRAMES: int = struct.field(pytree_node=False, default=30)
    ## Ending: the seabed hills stop, a sunken base scrolls in with the
    ## world, and once it reaches the trigger column a green objective
    ## appears above it, nearly stationary. Touching the objective freezes
    ## the scene for the usual colour cycle, pays the 5000 bonus and ends
    ## the game (the video's title screen still showed 3 lives).
    WC_BASE_WIDTH: int = struct.field(pytree_node=False, default=31)
    WC_BASE_HEIGHT: int = struct.field(pytree_node=False, default=26)
    WC_BASE_Y: int = struct.field(pytree_node=False, default=154)       ## sits on the water body bottom (rows 154-179)
    WC_GOAL_WIDTH: int = struct.field(pytree_node=False, default=8)
    WC_GOAL_HEIGHT: int = struct.field(pytree_node=False, default=18)
    WC_GOAL_Y: int = struct.field(pytree_node=False, default=135)
    WC_GOAL_TRIGGER_X: int = struct.field(pytree_node=False, default=66)   ## base column at which the objective appears
    WC_GOAL_OFFSET_X: int = struct.field(pytree_node=False, default=20)    ## objective column relative to the base's left edge
    WC_GOAL_DRIFT_PERIOD: int = struct.field(pytree_node=False, default=20) ## objective drifts left 1px every N frames
    ## Helicopter bombs keep sinking under the surface here (seen reaching
    ## the submarine's depth before fading), instead of dying at the waterline.
    WC_BOMB_FLOOR: int = struct.field(pytree_node=False, default=140)
    WIN_ANIMATION_FRAMES: int = struct.field(pytree_node=False, default=59) ## same freeze + colour cycle as a death

    ## Water scene: the satellite stops using the kitchen timer and instead
    ## releases its laser when it passes directly above the player (measured:
    ## the drop column always matched the player column). Aim once, straight
    ## down, no homing -- same as the stage one laser fall.
    ## Nailed with RAM-injection scans: at discrete check moments the
    ## satellite drops iff the drop column is 1..95px to the RIGHT of the
    ## player's hull -- it never fires while still left of the player, and
    ## drops opportunistically any time its belly is ahead of the boat.
    SATELLITE_DROP_AHEAD_MIN: int = struct.field(pytree_node=False, default=1)
    SATELLITE_DROP_AHEAD_MAX: int = struct.field(pytree_node=False, default=95)
    SATELLITE_CHECK_PERIOD: int = struct.field(pytree_node=False, default=30) ## check moments ~10-60f apart in ALE
    SATELLITE_WATER_MAX_DROPS: int = struct.field(pytree_node=False, default=2) ## 1-2 per pass observed
    SATELLITE_RESPAWN_FRAMES: int = struct.field(pytree_node=False, default=46) ## measured 44-48 frame gap
    SATELLITE_INITIAL_SPAWN_DELAY: int = struct.field(pytree_node=False, default=180) ## First satellite pass is delayed 180 frames after game start

    REWARD_STEP: float = struct.field(pytree_node=False, default=0.0)
    REWARD_DIAMOND: float = struct.field(pytree_node=False, default=1.0)
    REWARD_ENEMY: float = struct.field(pytree_node=False, default=2.0)
    REWARD_HIT_ENEMY: float = struct.field(pytree_node=False, default=-1.0) ## TODO: REMOVE
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
    hit_cooldown: chex.Array ## TODO: What for?
    death_timer: chex.Array ## frames left in the freeze-everything death animation
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
    satellite_respawn_timer: chex.Array ## gap between satellite passes
    ## Water scene scuba diver, one at a time like every other object
    scuba_x: chex.Array
    scuba_y: chex.Array
    scuba_active: chex.Array
    scuba_age: chex.Array ## frames since he entered; he vanishes on a clock
    scuba_respawn_timer: chex.Array ## breather before the next diver enters
    scuba_radioactive: chex.Array ## the diver currently holds the radioactive state
    scuba_radioactive_age: chex.Array ## how long he has been glowing
    ## Laser splash explosion: static in world space, rides the scroll
    splash_x: chex.Array
    splash_active: chex.Array
    splash_age: chex.Array
    ## Oil rig
    oil_rig_x: chex.Array
    oil_rig_y: chex.Array
    oil_rig_active: chex.Array
    oil_rig_visible: chex.Array  ## flash-only: rig is DRAWN only during the flash frames
    oil_rig_done: chex.Array     ## latch: rig already appeared this water scene (blocks re-trigger)
    stage1_start_step: chex.Array   ## step_count at the moment the water scene (stage 1) began
    oil_rig_seq: chex.Array      ## countdown driving the appear-right / gap / appear-left sequence (0 = idle)
    diamond_shot: chex.Array     ## True the frame a diamond is shot; triggers the rig next frame
    ## Second and third water scene roster. The rocket (and its debris)
    ## has WC_ROCKET_SLOTS fixed slots: water B only ever fills the first,
    ## the daylight scene runs both.
    rocket_x: chex.Array
    rocket_y: chex.Array
    rocket_active: chex.Array
    rocket_age: chex.Array ## ignites and launches after idling
    rocket_launch_age: chex.Array ## per-slot age at which the pad launches
    rocket_timer: chex.Array
    submarine_x: chex.Array
    submarine_active: chex.Array
    submarine_timer: chex.Array
    wb_heli_x: chex.Array
    wb_heli_active: chex.Array
    wb_heli_timer: chex.Array
    wb_ball_hits: chex.Array ## pink balls shot this scene (water B exit counter)
    ## Rocket explosion debris (the two red bars); timer is its age, y
    ## only moves in the daylight scene where the bars fall back down
    wb_flyer_x: chex.Array
    wb_flyer_y: chex.Array
    wb_flyer_active: chex.Array
    wb_flyer_timer: chex.Array
    ## Submarine's double-dot shot (water B and C)
    sub_torp_x: chex.Array
    sub_torp_y: chex.Array
    sub_torp_active: chex.Array
    sub_fired: chex.Array ## this pass already fired its one shot
    ## Daylight scene: the steamship on the waterline, the sunken base
    ## that ends the scene and the objective hovering above it
    ship_x: chex.Array
    ship_active: chex.Array
    ship_timer: chex.Array
    base_x: chex.Array
    base_active: chex.Array
    goal_x: chex.Array
    goal_active: chex.Array
    win_timer: chex.Array ## frames left in the mission-complete freeze
    stage_start_step: chex.Array ## step_count when the current scene began
    fired_bullet: chex.Array
    key: chex.PRNGKey


@struct.dataclass
class JamesBondObservation:
    """Object-centric observation matching observation_space()."""

    player: ObjectObservation
    diamonds: ObjectObservation
    helicopters: ObjectObservation
    satellites: ObjectObservation
    scubas: ObjectObservation
    ## rockets, submarine, pink helicopter, steamship, debris (water B and C)
    waterb_enemies: ObjectObservation
    ## sunken base and objective at the end of the daylight scene
    waterc_landmarks: ObjectObservation
    bullets: ObjectObservation
    lives: jnp.ndarray
    score: jnp.ndarray
    stage: jnp.ndarray


@struct.dataclass
class JamesBondInfo:
    """Debug/event info for smoke tests and future gameplay systems."""

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
        if consts is None:
            ## JB_START_STAGE lets playtesters jump straight into a later
            ## scene through scripts/play.py without touching code
            start_stage = int(os.environ.get("JB_START_STAGE", "1"))
            consts = JamesBondConstants(START_STAGE=min(max(start_stage, 0), 3))
        super().__init__(consts)
        self.renderer = JamesBondRenderer(self.consts)

    def reset(
        self, key: chex.PRNGKey = jax.random.PRNGKey(0)
    ) -> Tuple[JamesBondObservation, JamesBondState]:
        """Create an empty level state with inactive object slots."""

        if key is None:
            key = jax.random.PRNGKey(0)
        state_key, _ = jax.random.split(key)
        slots = self.consts.WC_ROCKET_SLOTS

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
            hit_cooldown=jnp.array(0, dtype=jnp.int32),
            death_timer=jnp.array(0, dtype=jnp.int32),
            diamond_x=jnp.array(0, dtype=jnp.int32),
            diamond_y=jnp.array(0, dtype=jnp.int32),
            diamond_active=jnp.array(0, dtype=jnp.bool_),
            pit_x=jnp.array(0, dtype=jnp.int32),
            pit_y=jnp.array(0, dtype=jnp.int32),
            pit_active=jnp.array(False, dtype=jnp.bool_),
            spawn_diamond_next=jnp.array(False, dtype=jnp.bool_), ## TODO: In state requires this, but is this array or zero-dimensional? Setting False here will make helicopter spawn first, which is true?
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
            satellite_respawn_timer=jnp.array(self.consts.SATELLITE_INITIAL_SPAWN_DELAY, dtype=jnp.int32),
            scuba_x=jnp.array(-1, dtype=jnp.int32),
            scuba_y=jnp.array(-1, dtype=jnp.int32),
            scuba_active=jnp.array(False, dtype=jnp.bool_),
            scuba_age=jnp.array(0, dtype=jnp.int32),
            scuba_respawn_timer=jnp.array(0, dtype=jnp.int32),
            scuba_radioactive=jnp.array(False, dtype=jnp.bool_),
            scuba_radioactive_age=jnp.array(0, dtype=jnp.int32),
            splash_x=jnp.array(-1, dtype=jnp.int32),
            splash_active=jnp.array(False, dtype=jnp.bool_),
            splash_age=jnp.array(0, dtype=jnp.int32),
            oil_rig_x=jnp.array(-1, dtype=jnp.int32),
            oil_rig_y=jnp.array(-1, dtype=jnp.int32),
            oil_rig_active=jnp.array(False, dtype=jnp.bool_),
            oil_rig_visible=jnp.array(False, dtype=jnp.bool_),
            oil_rig_done=jnp.array(False, dtype=jnp.bool_),
            stage1_start_step=jnp.array(0, dtype=jnp.int32),
            oil_rig_seq=jnp.array(0, dtype=jnp.int32),
            diamond_shot=jnp.array(False, dtype=jnp.bool_),
            #oil_rig_visible_timer=jnp.array(0, dtype=jnp.int32),
            rocket_x=jnp.full((slots,), -1, dtype=jnp.int32),
            rocket_y=jnp.full((slots,), -1, dtype=jnp.int32),
            rocket_active=jnp.zeros((slots,), dtype=jnp.bool_),
            rocket_age=jnp.zeros((slots,), dtype=jnp.int32),
            rocket_launch_age=jnp.full((slots,), self.consts.ROCKET_IGNITE_AGE, dtype=jnp.int32),
            rocket_timer=jnp.zeros((slots,), dtype=jnp.int32),
            submarine_x=jnp.array(-1, dtype=jnp.int32),
            submarine_active=jnp.array(False, dtype=jnp.bool_),
            submarine_timer=jnp.array(0, dtype=jnp.int32),
            wb_heli_x=jnp.array(-1, dtype=jnp.int32),
            wb_heli_active=jnp.array(False, dtype=jnp.bool_),
            wb_heli_timer=jnp.array(0, dtype=jnp.int32),
            wb_ball_hits=jnp.array(0, dtype=jnp.int32),
            wb_flyer_x=jnp.full((slots,), -1, dtype=jnp.int32),
            wb_flyer_y=jnp.full((slots,), self.consts.WB_FLYER_Y, dtype=jnp.int32),
            wb_flyer_active=jnp.zeros((slots,), dtype=jnp.bool_),
            wb_flyer_timer=jnp.zeros((slots,), dtype=jnp.int32),
            sub_torp_x=jnp.array(-1, dtype=jnp.int32),
            sub_torp_y=jnp.array(-1, dtype=jnp.int32),
            sub_torp_active=jnp.array(False, dtype=jnp.bool_),
            sub_fired=jnp.array(False, dtype=jnp.bool_),
            ship_x=jnp.array(-1, dtype=jnp.int32),
            ship_active=jnp.array(False, dtype=jnp.bool_),
            ship_timer=jnp.array(0, dtype=jnp.int32),
            base_x=jnp.array(-1, dtype=jnp.int32),
            base_active=jnp.array(False, dtype=jnp.bool_),
            goal_x=jnp.array(-1, dtype=jnp.int32),
            goal_active=jnp.array(False, dtype=jnp.bool_),
            win_timer=jnp.array(0, dtype=jnp.int32),
            stage_start_step=jnp.array(0, dtype=jnp.int32),
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

        def live_step(state: JamesBondState) -> JamesBondState:
            state = state.replace(
                ## The frame counter only ticks while the scene is alive:
                ## the real game freezes its clock during the death
                ## animation too, and every render animation and scroll
                ## offset derives from this counter.
                step_count=state.step_count + 1,
                hit_cooldown=jnp.maximum(state.hit_cooldown - 1, 0),
                fired_bullet=atari_action == Action.FIRE,
            )
            state = self._update_stage(state)
            state = self._step_player(state, atari_action)
            state = self._update_objects(state)
            state = self._update_enemy_bombs(state)
            state = self._resolve_collisions(state)
            return state

        def frozen_step(state: JamesBondState) -> JamesBondState:
            """The death animation: the whole scene stands still while the
            sprite color-cycles; when the timer runs out the player
            respawns and the scene is swept clean, exactly like the
            measured 59-frame freeze in the real game."""

            death_timer = state.death_timer - 1
            respawn = death_timer <= 0

            def sweep(v, park):
                return jnp.where(respawn, jnp.array(park, dtype=v.dtype), v)

            return state.replace(
                death_timer=death_timer,
                player_x=sweep(state.player_x, self.consts.PLAYER_INIT_X),
                player_y=sweep(state.player_y, self.consts.PLAYER_INIT_Y),
                player_jumping=sweep(state.player_jumping, False),
                player_falling=sweep(state.player_falling, False),
                player_fast_falling=sweep(state.player_fast_falling, False),
                player_in_air_step=sweep(state.player_in_air_step, 0),
                player_diving=sweep(state.player_diving, False),
                player_floating=sweep(state.player_floating, False),
                player_fast_floating=sweep(state.player_fast_floating, False),
                player_in_water_step=sweep(state.player_in_water_step, 0),
                ## Every actor and projectile leaves with the fallen agent.
                ## The shots also park their coordinates: their logic does
                ## not run while inactive, so a stale position would
                ## resurrect the old shot on the first post-respawn FIRE.
                player_bullet_active=sweep(state.player_bullet_active, False),
                player_bullet_x=sweep(state.player_bullet_x, -1),
                player_bullet_y=sweep(state.player_bullet_y, -1),
                player_bullet_step=sweep(state.player_bullet_step, -1),
                player_wbullet_active=sweep(state.player_wbullet_active, False),
                player_wbullet_x=sweep(state.player_wbullet_x, -1),
                player_wbullet_y=sweep(state.player_wbullet_y, -1),
                player_wbullet_step=sweep(state.player_wbullet_step, -1),
                helicopter_active=sweep(state.helicopter_active, False),
                helicopter_melee_step=sweep(state.helicopter_melee_step, 0),
                helicopter_bomb_active=sweep(state.helicopter_bomb_active, False),
                satellite_active=sweep(state.satellite_active, False),
                satellite_laser_active=sweep(state.satellite_laser_active, False),
                diamond_active=sweep(state.diamond_active, False),
                scuba_active=sweep(state.scuba_active, False),
                splash_active=sweep(state.splash_active, False),
                rocket_active=sweep(state.rocket_active, False),
                submarine_active=sweep(state.submarine_active, False),
                wb_heli_active=sweep(state.wb_heli_active, False),
                wb_flyer_active=sweep(state.wb_flyer_active, False),
                sub_torp_active=sweep(state.sub_torp_active, False),
                ship_active=sweep(state.ship_active, False),
                ## The pit resets to its measured post-death position
                pit_x=sweep(state.pit_x, 124),
                hit_cooldown=sweep(state.hit_cooldown, 0),
            )

        def win_step(state: JamesBondState) -> JamesBondState:
            """Scene cleared: the scene freezes for the same colour cycle
            as a death, then the bonus pays out and the next scene starts
            with a swept roster (after the last scene, stage 4 is the
            terminal marker, see _get_done)."""

            win_timer = state.win_timer - 1
            finished = win_timer <= 0
            return state.replace(
                win_timer=win_timer,
                score=jnp.where(
                    finished,
                    state.score + self.consts.SCORE_STAGE_BONUS,
                    state.score,
                ).astype(jnp.int32),
                stage=jnp.where(finished, state.stage + 1, state.stage).astype(jnp.int32),
                stage_start_step=jnp.where(finished, state.step_count, state.stage_start_step),
                **self._scene_clear_fields(state, finished),
            )

        state = jax.lax.cond(
            state.win_timer > 0,
            win_step,
            lambda s: jax.lax.cond(s.death_timer > 0, frozen_step, live_step, s),
            state,
        )

        _, next_key = jax.random.split(state.key)
        state = state.replace(key=next_key)

        observation = self._get_observation(state)
        reward = self._get_reward(previous_state, state)
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
                "helicopters": spaces.get_object_space(
                    n=None, screen_size=screen_size
                ),
                "satellites": spaces.get_object_space(
                    n=None, screen_size=screen_size
                ),
                ## Water scene scuba diver, single object like the enemies
                "scubas": spaces.get_object_space(
                    n=None, screen_size=screen_size
                ),
                ## two rockets, submarine, pink helicopter, steamship, two debris
                "waterb_enemies": spaces.get_object_space(
                    n=3 + 2 * self.consts.WC_ROCKET_SLOTS, screen_size=screen_size
                ),
                ## sunken base and objective (daylight scene ending)
                "waterc_landmarks": spaces.get_object_space(
                    n=2, screen_size=screen_size
                ),
                ## player air bullet, player water bullet, helicopter bomb,
                ## satellite laser, submarine shot
                "bullets": spaces.get_object_space(
                    n=5, screen_size=screen_size
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
        ## Water B / C roster in one fixed-slot group: the rocket slots,
        ## submarine, pink helicopter, steamship, then the debris slots.
        ## The shared box is the largest of the sprites.
        waterb_enemies = self._object_group_observation(
            jnp.concatenate([
                state.rocket_x,
                jnp.stack([state.submarine_x, state.wb_heli_x, state.ship_x]),
                state.wb_flyer_x,
            ]),
            jnp.concatenate([
                state.rocket_y,
                jnp.array([
                    self.consts.SUBMARINE_Y,
                    self.consts.WB_HELI_Y,
                    self.consts.WC_SHIP_Y,
                ], dtype=jnp.int32),
                state.wb_flyer_y,
            ]),
            jnp.concatenate([
                state.rocket_active,
                jnp.stack([state.submarine_active, state.wb_heli_active, state.ship_active]),
                state.wb_flyer_active,
            ]),
            self.consts.SUBMARINE_WIDTH,
            self.consts.SUBMARINE_HEIGHT,
        )
        waterc_landmarks = self._object_group_observation(
            jnp.stack([state.base_x, state.goal_x]),
            jnp.array([self.consts.WC_BASE_Y, self.consts.WC_GOAL_Y], dtype=jnp.int32),
            jnp.stack([state.base_active, state.goal_active]),
            self.consts.WC_BASE_WIDTH,
            self.consts.WC_BASE_HEIGHT,
        )
        ## All projectiles in one group (the 1x4 box is close enough for
        ## the submarine's 2x5 shot too): player air bullet, player water
        ## bullet, helicopter bomb, satellite laser, submarine shot
        bullets = self._object_group_observation(
            jnp.stack([
                state.player_bullet_x,
                state.player_wbullet_x,
                state.helicopter_bomb_x,
                state.satellite_laser_x,
                state.sub_torp_x,
            ]),
            jnp.stack([
                state.player_bullet_y,
                state.player_wbullet_y,
                state.helicopter_bomb_y,
                state.satellite_laser_y,
                state.sub_torp_y,
            ]),
            jnp.stack([
                state.player_bullet_active,
                state.player_wbullet_active,
                state.helicopter_bomb_active,
                state.satellite_laser_active,
                state.sub_torp_active,
            ]),
            self.consts.BULLET_WIDTH,
            self.consts.BULLET_HEIGHT,
        )
        return JamesBondObservation(
            player=player,
            diamonds=diamonds,
            helicopters=helicopters,
            satellites=satellites,
            scubas=scubas,
            waterb_enemies=waterb_enemies,
            waterc_landmarks=waterc_landmarks,
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
            fired_bullet=state.fired_bullet,
            score=state.score,
            lives=state.lives,
            stage=state.stage,
            step_count=state.step_count,
        )

    def _decode_action(self, action: chex.Array) -> chex.Array:
        """Translate compact action-space indices to JAXAtariAction values."""

        return jnp.take(self.ACTION_SET, jnp.asarray(action, dtype=jnp.int32))

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
        
        ## Diving starts on DOWN (the mirror of the air logic, where UP
        ## starts the jump). The old copy used up_pressed here, but the
        ## stage router only sends DOWN presses into the water branch, so
        ## the dive could never begin.
        player_diving = jnp.where(
            player_diving,
            player_diving,
            jnp.where(
                jnp.logical_and(
                    jnp.logical_and(down_pressed, player_in_water_step < 71),
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
        
        ## UP while under water rushes the boat back to the surface, the
        ## mirror of DOWN fast-falling a jump in the air logic.
        player_fast_floating = jnp.where(
            player_fast_floating,
            True,
            jnp.where(
                jnp.logical_and(
                    up_pressed,
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
                    jnp.where(player_wbullet_step < 8,
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
                    state.player_wbullet_step >= 8, ## TODO: Maybe more?
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

    def _step_player(
        self, state: JamesBondState, atari_action: chex.Array
    ) -> JamesBondState:

        return jax.lax.switch( ## Stage indexing starts with 0
            state.stage,
            [
                self.step_player_stage_one,
                self.step_player_stage_two,
                ## The second and third water scenes drive the same boat
                ## physics and fire the same two rounds (the daylight
                ## video shows the anti-air shot and the depth charge
                ## in the air together).
                self.step_player_stage_two,
                self.step_player_stage_two,
            ],
            (state, atari_action)
        )

    def _update_stage(self, state: JamesBondState) -> JamesBondState: ## TODO: change text, delete unneccessary variables
        """Advance the scene clock and roll over to the next scene when due.

        The real game cycles terrain types as the vehicle travels; here the
        land scene (stage 0) runs STAGE_ONE_LENGTH frames, then the water
        scene (stage 1) STAGE_TWO_LENGTH frames, then back to land. On a
        switch every scene-owned object is cleared so the new scene starts
        empty, exactly like the terrain handover in the original.
        """

        lives_lost = self.consts.MAX_LIVES - state.lives

        ## Land -> water A on the clock (each lost life stretches it).
        ## The old rule was not stage-aware and dragged water B straight
        ## back to water A one frame after the rig landing.
        land_done = jnp.logical_and(
            state.stage == 0,
            state.step_count > 3000 + lives_lost * 1000,
        )

        ## Landing on TOP of the oil rig ends the water scene and advances
        ## to the next stage (real-game level-end condition). Side contact is
        ## handled separately as a lethal collision in _resolve_oil_rig_collision.
        rig_over_deck = jnp.logical_and(
            state.player_x + self.consts.PLAYER_COLLISION_WIDTH > state.oil_rig_x,
            state.player_x < state.oil_rig_x + self.consts.OIL_RIG_WIDTH,
        )
        rig_on_top = jnp.logical_and(
            state.player_y + self.consts.PLAYER_COLLISION_HEIGHT >= state.oil_rig_y,
            state.player_y <= state.oil_rig_y + self.consts.OIL_RIG_TOP_LAND_MARGIN,
        )
        landed_on_rig = state.oil_rig_active & rig_over_deck & rig_on_top
        water_a_done = jnp.logical_and(state.stage == 1, landed_on_rig)

        ## Water B ends by shooting the pink balls and water C by diving
        ## into the objective; both run through the scene-clear freeze
        ## (win_step), which advances the stage itself.
        new_stage = jnp.where(
            land_done | water_a_done,
            state.stage + 1,
            state.stage,
        ).astype(jnp.int32)

        switch = state.stage != new_stage

        stage_start_step = jnp.where(switch, state.step_count, state.stage_start_step)

        ## Remember when the water scene (stage 1) begins, so the oil rig
        ## can be time-gated to appear only 1000+ steps into it.
        entering_stage1 = jnp.logical_and(switch, new_stage == 1)
        stage1_start_step = jnp.where(
            entering_stage1, state.step_count, state.stage1_start_step
        )
        oil_rig_done_reset = jnp.where(entering_stage1, jnp.array(False, dtype=jnp.bool_), state.oil_rig_done)
        return state.replace(
            stage=new_stage,
            stage_start_step=stage_start_step,
            stage1_start_step=stage1_start_step,
            oil_rig_done=oil_rig_done_reset,
            **self._scene_clear_fields(state, switch),
        )

    def _scene_clear_fields(self, state: JamesBondState, switch: chex.Array) -> dict:
        """Every scene-owned actor, projectile and counter, parked when
        `switch` is set -- shared by the scroll-driven handovers in
        _update_stage and by the scene-clear freeze in step()."""

        def clear(v, park):
            return jnp.where(switch, jnp.array(park, dtype=v.dtype), v)

        return dict(
            ## Water B / C actors do not carry over
            rocket_active=clear(state.rocket_active, False),
            rocket_timer=clear(state.rocket_timer, 0),
            submarine_active=clear(state.submarine_active, False),
            submarine_timer=clear(state.submarine_timer, 0),
            wb_heli_active=clear(state.wb_heli_active, False),
            wb_ball_hits=clear(state.wb_ball_hits, 0),
            wb_flyer_active=clear(state.wb_flyer_active, False),
            sub_torp_active=clear(state.sub_torp_active, False),
            sub_fired=clear(state.sub_fired, False),
            ship_active=clear(state.ship_active, False),
            ship_timer=clear(state.ship_timer, 0),
            base_active=clear(state.base_active, False),
            goal_active=clear(state.goal_active, False),
            ## Land objects vanish at the shoreline
            helicopter_active=clear(state.helicopter_active, False),
            helicopter_melee_step=clear(state.helicopter_melee_step, 0),
            diamond_active=clear(state.diamond_active, False),
            pit_active=clear(state.pit_active, False),
            helicopter_bomb_active=clear(state.helicopter_bomb_active, False),
            helicopter_bombs_dropped=clear(state.helicopter_bombs_dropped, 0),
            helicopter_bomb_timer=clear(state.helicopter_bomb_timer, 0),
            ## The satellite and its laser reset for the new scene
            satellite_active=clear(state.satellite_active, False),
            satellite_laser_active=clear(state.satellite_laser_active, False),
            satellite_laser_timer=clear(
                state.satellite_laser_timer, self.consts.SATELLITE_LASER_DROP_PERIOD
            ),
            satellite_lasers_dropped=clear(state.satellite_lasers_dropped, 0),
            ## Water objects vanish when the water ends
            scuba_active=clear(state.scuba_active, False),
            scuba_age=clear(state.scuba_age, 0),
            scuba_radioactive=clear(state.scuba_radioactive, False),
            scuba_radioactive_age=clear(state.scuba_radioactive_age, 0),
            scuba_respawn_timer=clear(
                state.scuba_respawn_timer, self.consts.SCUBA_RESPAWN_FRAMES
            ),
            splash_active=clear(state.splash_active, False),
            splash_age=clear(state.splash_age, 0),
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
        ## Measured in ALE: the diver swims left 1px every 4th frame (0.25
        ## px per frame) at a fixed depth, never chases, and vanishes
        ## mid-screen on an age clock -- he does not swim off the left
        ## edge (two clean episodes lasted exactly 333 frames each).
        next_scuba_x = jnp.where(
            state.step_count % 4 == 0,
            state.scuba_x - 1,
            state.scuba_x
        )
        next_scuba_y = state.scuba_y
        scuba_age = jnp.where(state.scuba_active, state.scuba_age + 1, 0)
        ## An existing glow runs on its own clock. Once its fixed lifetime
        ## expires it may be activated again only by player proximity.
        scuba_rad_age = jnp.where(
            state.scuba_radioactive, state.scuba_radioactive_age + 1, 0
        )
        still_glowing = state.scuba_radioactive & (
            scuba_rad_age < self.consts.SCUBA_RADIOACTIVE_FRAMES
        )
        ## A radioactive diver remains until his glow has finished, even if
        ## his ordinary on-screen lifetime expires first.
        next_scuba_active = state.scuba_active & (
            (scuba_age < self.consts.SCUBA_LIFETIME_FRAMES) | still_glowing
        )
        ## The diver becomes radioactive as soon as the boat is horizontally
        ## close enough. This is independent of every satellite projectile.
        ## Measure the visible edge-to-edge gap rather than the two sprites'
        ## left origins, so a diver that looks close really is in range.
        player_right = state.player_x + self.consts.PLAYER_WIDTH
        scuba_right = next_scuba_x + self.consts.SCUBA_WIDTH
        scuba_horizontal_gap = jnp.maximum(
            jnp.maximum(
                next_scuba_x - player_right,
                state.player_x - scuba_right,
            ),
            0,
        )
        scuba_in_range = (
            (state.stage == 1)
            & next_scuba_active
            & (scuba_horizontal_gap <= self.consts.SCUBA_RADIOACTIVE_RANGE)
        )
        newly_radioactive = scuba_in_range & (~state.scuba_radioactive)
        next_scuba_radioactive = (
            (still_glowing | scuba_in_range) & next_scuba_active
        )
        scuba_rad_age = jnp.where(newly_radioactive, 0, scuba_rad_age)

        # Laser splash explosion (rides the world scroll)
        ## Static in world space: it drifts left with the terrain, 1px
        ## every 4th frame, and burns out on its own clock.
        next_splash_x = jnp.where(
            state.step_count % 4 == 0,
            state.splash_x - 1,
            state.splash_x
        )
        splash_age = jnp.where(state.splash_active, state.splash_age + 1, 0)
        next_splash_active = state.splash_active & (
            splash_age < self.consts.SPLASH_LIFETIME_FRAMES
        )

        # Oil rig: stationary. Active purely as a frame window -- it is
        ## glimpsed in the sky flash for a few frames, then gone. Computed
        ## directly from step_count so spawn/despawn can't miss each other.
        next_oil_rig_y = state.oil_rig_y
        ## Oil rig is triggered by SHOOTING THE DIAMOND (not a frame timer).
        ## diamond_shot (set last frame in the diamond collision) starts the
        ## sequence: appear on the RIGHT with a flash, disappear, then reappear
        ## on the LEFT near the player. Only (re)start when idle (seq == 0).
        in_water = state.stage == 1
        steps_into_stage1 = state.step_count - state.stage1_start_step
        past_delay = steps_into_stage1 >= self.consts.OIL_RIG_MIN_STAGE1_STEPS
        diamond_shot_any = jnp.any(state.diamond_shot)
        start_seq = jnp.logical_and(
            jnp.logical_and(
                jnp.logical_and(diamond_shot_any, in_water),
                jnp.logical_and(past_delay, jnp.logical_not(state.oil_rig_done)),
            ),
            state.oil_rig_seq == 0,
        )
        oil_rig_seq = jnp.where(
            start_seq,
            jnp.array(self.consts.OIL_RIG_SEQ_TOTAL, dtype=jnp.int32),
            jnp.maximum(state.oil_rig_seq - 1, 0),
        )
        oil_rig_seq = jnp.where(state.death_timer > 0, jnp.array(0, dtype=jnp.int32), oil_rig_seq)
        oil_rig_done = jnp.logical_or(state.oil_rig_done, start_seq)
        ## Phase boundaries (counting DOWN from OIL_RIG_SEQ_TOTAL):
        ##   TOTAL .. RIGHT_END  -> visible on the RIGHT (+ flash)
        ##   RIGHT_END .. LEFT_START -> hidden (the disappear gap)
        ##   LEFT_START .. 1     -> visible on the LEFT, near the player
        on_right = jnp.logical_and(oil_rig_seq <= self.consts.OIL_RIG_SEQ_TOTAL,
                                   oil_rig_seq > self.consts.OIL_RIG_SEQ_RIGHT_END)
        on_left = jnp.logical_and(oil_rig_seq <= self.consts.OIL_RIG_SEQ_LEFT_START,
                                  oil_rig_seq > 0)
        next_oil_rig_active = jnp.logical_or(on_right, on_left)
        ## Drawn ONLY during flash frames (first STRIKE_LEN of each phase),
        ## so the move between right and left is never seen (no teleport look).
        _sl = self.consts.OIL_RIG_STRIKE_LEN
        _fr = jnp.logical_and(oil_rig_seq <= self.consts.OIL_RIG_SEQ_TOTAL,
                              oil_rig_seq > self.consts.OIL_RIG_SEQ_TOTAL - _sl)
        _fl = jnp.logical_and(oil_rig_seq <= self.consts.OIL_RIG_SEQ_LEFT_START,
                              oil_rig_seq > self.consts.OIL_RIG_SEQ_LEFT_START - _sl)
        next_oil_rig_visible = jnp.logical_and(next_oil_rig_active, jnp.logical_or(_fr, _fl))
        ## During the RIGHT phase the rig is visible and drifts a little to
        ## the left before it disappears. Progress through the right phase:
        ##   0 at the start (seq == TOTAL) .. 1 at the end (seq == RIGHT_END)
        right_span = jnp.maximum(self.consts.OIL_RIG_SEQ_TOTAL - self.consts.OIL_RIG_SEQ_RIGHT_END, 1)
        right_prog = (self.consts.OIL_RIG_SEQ_TOTAL - oil_rig_seq).astype(jnp.int32)
        right_x = self.consts.OIL_RIG_RIGHT_X - (right_prog * self.consts.OIL_RIG_RIGHT_SLIDE) // right_span
        next_oil_rig_x = jnp.where(
            on_right,
            right_x.astype(jnp.int32),
            jnp.where(on_left,
                      jnp.array(self.consts.OIL_RIG_LEFT_X, dtype=jnp.int32),
                      state.oil_rig_x),
        )

        # Second and third water scene roster (stages 2 and 3)
        in_water_b = state.stage == 2
        in_water_c = state.stage == 3
        in_water_bc = in_water_b | in_water_c
        scroll_tick = state.step_count % 4 == 0
        frames_in_stage = state.step_count - state.stage_start_step
        ## Daylight phases: rockets first, ships later (see WC_SHIP_PHASE_START)
        rocket_phase = in_water_c & (frames_in_stage < self.consts.WC_SHIP_PHASE_START)
        ship_phase = in_water_c & (frames_in_stage >= self.consts.WC_SHIP_PHASE_START)
        ## Rocket, measured cycle: floats submerged riding the world scroll
        ## until its launch age, then climbs straight up 1px/frame and
        ## bursts with its tip at the measured explosion row -- leaving
        ## the red debris bars (and, in water B, a short full-sky flash).
        ## Every rocket variable holds one entry per launch-pad slot.
        rocket_age = jnp.where(state.rocket_active, state.rocket_age + 1, 0)
        rocket_flying = rocket_age >= state.rocket_launch_age
        ## Water B's rocket goes straight up; the daylight one keeps
        ## drifting with the world while it climbs (video: -3px / 12f).
        rocket_drifts = state.rocket_active & scroll_tick & ((~rocket_flying) | in_water_c)
        next_rocket_x = jnp.where(rocket_drifts, state.rocket_x - 1, state.rocket_x)
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
        ## Submarine (water B and C): enters from the LEFT and cruises
        ## right, 2px every 3 frames (video: ~0.6-0.7 px/f in both scenes)
        sub_dx = jnp.where(state.step_count % 3 != 0, 1, 0)
        next_submarine_x = jnp.where(
            state.submarine_active,
            state.submarine_x + sub_dx,
            state.submarine_x
        )
        next_submarine_active = state.submarine_active & (
            next_submarine_x < self.consts.OBJECT_EXIT_X
        )
        ## Submarine shot (water B and C, video): two stacked dots leave
        ## the bow as the submarine crosses SUB_FIRE_X, run diagonally
        ## back and up at (-2,-1)/frame until they sit just under the
        ## surface, then straight left along that row at 4px every 3
        ## frames until they leave the screen.
        at_level = state.sub_torp_y <= self.consts.SUB_SHOT_LEVEL_Y
        run_dx = jnp.where(state.step_count % 3 == 0, -2, -1)
        torp_dx = jnp.where(at_level, run_dx, -2)
        torp_dy = jnp.where(at_level, 0, -1)
        next_torp_x = jnp.where(state.sub_torp_active, state.sub_torp_x + torp_dx, -1)
        next_torp_y = jnp.where(
            state.sub_torp_active,
            jnp.maximum(state.sub_torp_y + torp_dy, self.consts.SUB_SHOT_LEVEL_Y),
            -1,
        )
        next_torp_active = state.sub_torp_active & (
            next_torp_x > self.consts.GAME_AREA_MIN_X - self.consts.SUB_SHOT_WIDTH
        )
        crossed_fire_x = jnp.logical_and(
            state.submarine_x < self.consts.SUB_FIRE_X,
            next_submarine_x >= self.consts.SUB_FIRE_X,
        )
        fire_torp = (
            in_water_bc & next_submarine_active & (~state.sub_fired)
            & (~next_torp_active) & crossed_fire_x
        )
        next_torp_x = jnp.where(
            fire_torp, next_submarine_x + self.consts.SUBMARINE_WIDTH - self.consts.SUB_SHOT_WIDTH, next_torp_x
        )
        next_torp_y = jnp.where(fire_torp, jnp.array(self.consts.SUBMARINE_Y - 3, dtype=jnp.int32), next_torp_y)
        next_torp_active = next_torp_active | fire_torp
        ## One shot per pass: the flag lifts when the submarine is gone
        next_sub_fired = (state.sub_fired | fire_torp) & next_submarine_active
        ## Pink ball: crosses the sky left to right, 7px every 4 frames
        ## (+2,+2,+2,+1), and leaves at the right edge if nobody pops it
        next_wb_heli_x = jnp.where(
            state.wb_heli_active,
            state.wb_heli_x + jnp.where(state.step_count % 4 == 3, 1, 2),
            state.wb_heli_x
        )
        next_wb_heli_active = state.wb_heli_active & (
            next_wb_heli_x < self.consts.OBJECT_EXIT_X
        )
        ## Rocket debris: the two red bars drift with the world and fall
        ## 1px/frame back to the waterline (the clock is held while
        ## falling), float there briefly as the sparkle, then go. Same
        ## lifecycle in water B and the daylight scene (team decision --
        ## the old ALE note had water B's bars lingering in the sky).
        debris_age = jnp.where(state.wb_flyer_active, state.wb_flyer_timer + 1, 0)
        next_wb_flyer_x = jnp.where(
            state.wb_flyer_active & scroll_tick,
            state.wb_flyer_x - 1,
            state.wb_flyer_x
        )
        debris_falling = in_water_bc & (state.wb_flyer_y < self.consts.WC_DEBRIS_REST_Y)
        next_wb_flyer_y = jnp.where(
            state.wb_flyer_active & debris_falling,
            state.wb_flyer_y + 1,
            state.wb_flyer_y
        )
        debris_age = jnp.where(debris_falling, 0, debris_age)
        next_wb_flyer_active = state.wb_flyer_active & (
            debris_age < self.consts.WC_DEBRIS_REST_FRAMES
        )
        ## The burst hands over: rocket out, debris in centred on the burst column
        next_wb_flyer_active = next_wb_flyer_active | rocket_explodes
        debris_offset = jnp.where(
            in_water_c,
            (self.consts.WC_ROCKET_WIDTH - self.consts.WB_FLYER_WIDTH) // 2,
            (self.consts.ROCKET_WIDTH - self.consts.WB_FLYER_WIDTH) // 2,
        )
        next_wb_flyer_x = jnp.where(rocket_explodes, next_rocket_x + debris_offset, next_wb_flyer_x)
        next_wb_flyer_y = jnp.where(
            rocket_explodes,
            jnp.array(self.consts.WB_FLYER_Y, dtype=jnp.int32),
            next_wb_flyer_y,
        )
        debris_age = jnp.where(rocket_explodes, 0, debris_age)

        ## Steamship (daylight scene): rides the waterline right to left at
        ## the helicopter's cruising speed, 3px every 5 frames
        ship_move = (state.step_count % 5 == 0) | (state.step_count % 5 == 2) | (state.step_count % 5 == 4)
        next_ship_x = jnp.where(
            state.ship_active & ship_move,
            state.ship_x - 1,
            state.ship_x
        )
        next_ship_active = state.ship_active & (
            next_ship_x > self.consts.GAME_AREA_MIN_X - self.consts.WC_SHIP_WIDTH
        )

        ## Sunken base (daylight ending): scrolls in from the right once
        ## the scene has run its course and rides the world scroll; the
        ## objective appears above it at the trigger column and then
        ## drifts left only very slowly, so the base passes underneath it.
        next_base_x = jnp.where(
            state.base_active & scroll_tick,
            state.base_x - 1,
            state.base_x
        )
        next_base_active = state.base_active & (
            next_base_x > self.consts.GAME_AREA_MIN_X - self.consts.WC_BASE_WIDTH
        )
        ## If the objective was missed and drifted away, the base comes
        ## round again so the mission can still be finished.
        spawn_base = (
            in_water_c & (~next_base_active) & (~state.goal_active)
            & (frames_in_stage >= self.consts.WC_LENGTH)
        )
        next_base_active = next_base_active | spawn_base
        next_base_x = jnp.where(
            spawn_base,
            jnp.array(self.consts.OBJECT_SPAWN_X_FAR, dtype=jnp.int32),
            next_base_x,
        )
        goal_drift = state.step_count % self.consts.WC_GOAL_DRIFT_PERIOD == 0
        next_goal_x = jnp.where(
            state.goal_active & goal_drift,
            state.goal_x - 1,
            state.goal_x
        )
        next_goal_active = state.goal_active & (
            next_goal_x > self.consts.GAME_AREA_MIN_X - self.consts.WC_GOAL_WIDTH
        )
        ## The base moves one column at a time, so the trigger column is
        ## crossed exactly once per pass
        spawn_goal = (
            in_water_c & next_base_active & (~next_goal_active)
            & (next_base_x == self.consts.WC_GOAL_TRIGGER_X)
        )
        next_goal_active = next_goal_active | spawn_goal
        next_goal_x = jnp.where(
            spawn_goal,
            next_base_x + self.consts.WC_GOAL_OFFSET_X,
            next_goal_x,
        )

        # Enemies
        ## Helicopter enemy (Scroll left)
        ## 1. Determine which speed zone the helicopter is currently in, also affecting melee behavior of helicopter
        ## (the daylight scene's helicopter crosses at a constant ~0.6 px/f
        ## with no searchlight sweep, so it never enters the slow zone there)
        in_slow_mode = (state.helicopter_x <= 96) & (state.helicopter_x > 63) & (~in_water_c)
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
        ## Measured exactly: +1,+1,+1,+0 repeating = 3px every 4 frames
        next_satellite_x = jnp.where(
            state.step_count % 4 != 3,
            state.satellite_x + 1,
            state.satellite_x
        )
        next_satellite_y = state.satellite_y
        ## The satellite crosses the whole screen (measured pass ~209
        ## frames, entry at the left edge, exit past the right edge)
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
        # Check whose turn it is to spawn. The pit is land-only; the
        # diamond floats through the sky of every scene (measured at
        # y60-64 on land and y62 over the water), and the red helicopter
        # also patrols the first water scene, dropping the same bombs
        # (measured: enters right, ~0.58 px/f slowing mid-screen, ~2 bombs
        # per crossing). Only the second water scene retires it.
        ## The daylight scene has no floating diamond, and its red
        ## helicopter comes back pass after pass (video), so there the
        ## helicopter ignores the turn flag.
        spawn_diamond = row_57_empty & (state.spawn_diamond_next | in_water_b) & (~in_water_c)
        ## The helicopter spawning will be delayed by 75 frames from initial state
        helicopter_delay_passed = state.step_count >= 75
        helicopter_turn = ((~state.spawn_diamond_next) & (~in_water_b)) | in_water_c
        spawn_helicopter = row_57_empty & helicopter_turn & (~state.oil_rig_active) & (helicopter_delay_passed)
        ## The satellite takes a measured ~65 frame breather between passes;
        ## the gap also lets its per-pass laser counter reset.
        satellite_respawn_timer = jnp.where(
            next_satellite_active,
            jnp.array(self.consts.SATELLITE_RESPAWN_FRAMES, dtype=jnp.int32),
            jnp.maximum(state.satellite_respawn_timer - 1, 0),
        )
        ## The satellite patrols the land and the first water scene; the
        ## second water scene never shows it (22k frame scan).
        ## The first satellite is delayed by 180 frames
        satellite_delay_passed = state.step_count > self.consts.SATELLITE_INITIAL_SPAWN_DELAY
        can_spawn_satellite = (
            (~next_satellite_active) & (satellite_respawn_timer == 0) & (satellite_delay_passed) & (state.stage <= 1) & (~state.oil_rig_active)
        )
        can_spawn_pit = (~next_pit_active) & on_land ## Only spawn when the previous pit left the screen

        ## Water B / C spawners: each object enters on its own staggered
        ## breather so the roster stays mixed. `scene` says which scenes
        ## the object belongs to; the timer is parked while it is absent.
        def waterb_spawner(active, timer, gap, scene):
            next_timer = jnp.where(
                active | (~scene),
                jnp.asarray(gap, dtype=jnp.int32),
                jnp.maximum(timer - 1, 0),
            )
            spawn = scene & (~active) & (next_timer == 0)
            return spawn, next_timer

        ## Rocket pads. Water B keeps its single pad (slot 0) appearing
        ## mid-screen (measured x~85) on the 256-frame cycle: ~85 frames
        ## of life + the breather. The daylight scene fills every slot at
        ## a random column, after a random breather, and lets each pad
        ## ride the seabed for a random while before it launches.
        slot_ids = jnp.arange(self.consts.WC_ROCKET_SLOTS)
        rocket_scene = (in_water_b & (slot_ids == 0)) | rocket_phase
        random_gap = jax.random.randint(
            jax.random.fold_in(state.key, 7), slot_ids.shape,
            self.consts.WC_ROCKET_GAP_MIN, self.consts.WC_ROCKET_GAP_MAX + 1,
        )
        random_x = jax.random.randint(
            jax.random.fold_in(state.key, 8), slot_ids.shape,
            self.consts.WC_ROCKET_SPAWN_MIN_X, self.consts.WC_ROCKET_SPAWN_MAX_X + 1,
        )
        random_idle = jax.random.randint(
            jax.random.fold_in(state.key, 9), slot_ids.shape,
            self.consts.WC_ROCKET_IDLE_MIN, self.consts.WC_ROCKET_IDLE_MAX + 1,
        )
        rocket_gap = jnp.where(in_water_c, random_gap, self.consts.ROCKET_RESPAWN_FRAMES)
        spawn_rocket, rocket_timer = waterb_spawner(
            next_rocket_active, state.rocket_timer, rocket_gap, rocket_scene)
        next_rocket_active = next_rocket_active | spawn_rocket
        next_rocket_x = jnp.where(
            spawn_rocket,
            jnp.where(in_water_c, random_x, self.consts.ROCKET_SPAWN_X),
            next_rocket_x,
        ).astype(jnp.int32)
        next_rocket_y = jnp.where(spawn_rocket, self.consts.ROCKET_Y, next_rocket_y).astype(jnp.int32)
        rocket_age = jnp.where(spawn_rocket, 0, rocket_age)
        rocket_launch_age = jnp.where(
            spawn_rocket,
            jnp.where(in_water_c, random_idle, self.consts.ROCKET_IGNITE_AGE),
            state.rocket_launch_age,
        ).astype(jnp.int32)

        ## The submarine enters from the left in both water scenes
        spawn_submarine, submarine_timer = waterb_spawner(
            next_submarine_active, state.submarine_timer,
            self.consts.SUBMARINE_RESPAWN_FRAMES, in_water_bc)
        next_submarine_active = next_submarine_active | spawn_submarine
        next_submarine_x = jnp.where(
            spawn_submarine, self.consts.SUB_SPAWN_X, next_submarine_x
        ).astype(jnp.int32)
        next_sub_fired = next_sub_fired & (~spawn_submarine)

        ## The pink ball belongs to water B only and enters at the LEFT
        ## edge; the daylight scene flies the red land helicopter instead
        ## (spawned above)
        spawn_wb_heli, wb_heli_timer = waterb_spawner(
            next_wb_heli_active, state.wb_heli_timer,
            self.consts.WB_HELI_RESPAWN_FRAMES, in_water_b)
        next_wb_heli_active = next_wb_heli_active | spawn_wb_heli
        next_wb_heli_x = jnp.where(
            spawn_wb_heli,
            self.consts.GAME_AREA_MIN_X - self.consts.WB_HELI_WIDTH,
            next_wb_heli_x,
        ).astype(jnp.int32)

        ## Steamship: the daylight scene's second phase only, back-to-back
        ## passes from the right
        spawn_ship, ship_timer = waterb_spawner(
            next_ship_active, state.ship_timer,
            self.consts.WC_SHIP_RESPAWN_FRAMES, ship_phase)
        next_ship_active = next_ship_active | spawn_ship
        next_ship_x = jnp.where(spawn_ship, self.consts.OBJECT_SPAWN_X_FAR, next_ship_x).astype(jnp.int32)

        ## The debris has no spawner of its own: it is born where the
        ## rocket bursts (handled above with the rocket lifecycle).
        wb_flyer_timer = debris_age

        ## Scuba diver: water only, one at a time, entering from the right
        ## edge with a breather between divers.
        scuba_respawn_timer = jnp.where(
            next_scuba_active | on_land,
            jnp.array(self.consts.SCUBA_RESPAWN_FRAMES, dtype=jnp.int32),
            jnp.maximum(state.scuba_respawn_timer - 1, 0),
        )
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
        # Check if either helicopter or diamond is spawning on this frame
        spawn_occurred = jnp.logical_or(spawn_diamond, spawn_helicopter)
        # Flip the turn flag ONLY if a spawn is happening on this frame
        next_spawn_diamond_next = jnp.where(
            spawn_occurred,
            ~state.spawn_diamond_next, ## Swap to the other object for next time
            state.spawn_diamond_next ## Keep it the same while they are flying
        )
        # Diamonds
        # Apply new active status, position coordinates for spawned diamonds
        next_diamond_active = next_diamond_active | spawn_diamond
        next_diamond_x = jnp.where(
            spawn_diamond,
            self.consts.OBJECT_SPAWN_X_FAR, ## enters at the right screen edge like in ALE
            next_diamond_x
        )
        next_diamond_y = jnp.where(
            spawn_diamond,
            ## Measured heights: gem core rows y60-64 on land, y62 over water
            jnp.where(on_land, 57, 62),
            next_diamond_y
        )
        # Oil rig
        ## Oil rig visibility countdown
        ##next_oil_rig_visible_timer = jnp.maximum(state.oil_rig_visible_timer - 1, 0)
        ## Trigger visibility flash when player scores on diamond or scuba
        ## Have to pass 'scored_points_this_frame' boolean from collision
        ##next_oil_rig_visible_timer = jnp.where(
        ##    scored_points_this_frame & state.oil_rig_active,
        ##    self.consts.OIL_RIG_FLASH_FRAMES, # Placeholder for N frames
        ##    next_oil_rig_visible_timer
        ##)
        # Position the rig at its fixed spot whenever the window (set above)
        ## has it active. The window alone owns active/inactive now.
        next_oil_rig_y = jnp.where(
            next_oil_rig_active,
            self.consts.OIL_RIG_Y,
            next_oil_rig_y
        )
        # Enemies
        ## Helicopter
        # Apply new active status, position coordinates for spawned helicopter enemies
        next_helicopter_active = next_helicopter_active | spawn_helicopter
        next_helicopter_x = jnp.where(
            spawn_helicopter,
            self.consts.OBJECT_SPAWN_X, ## measured entry around column 150
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
            self.consts.OBJECT_SPAWN_X_FAR, ## next pit enters right as the last leaves: ~160px world spacing
            next_pit_x
        )
        next_pit_y = jnp.where(
            can_spawn_pit,
            119, ## sprite row 0 = two rows above the crater rim; crater top lands just over the road
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
            scuba_radioactive=next_scuba_radioactive,
            scuba_radioactive_age=scuba_rad_age,
            scuba_respawn_timer=scuba_respawn_timer,
            splash_x=next_splash_x,
            splash_active=next_splash_active,
            splash_age=splash_age,
            oil_rig_x=next_oil_rig_x,
            oil_rig_y=next_oil_rig_y,
            oil_rig_active=next_oil_rig_active,
            oil_rig_visible=next_oil_rig_visible,
            oil_rig_done=oil_rig_done,
            oil_rig_seq=oil_rig_seq,
            stage1_start_step=state.stage1_start_step,
            rocket_x=next_rocket_x,
            rocket_y=next_rocket_y,
            rocket_active=next_rocket_active,
            rocket_age=rocket_age,
            rocket_launch_age=rocket_launch_age,
            rocket_timer=rocket_timer,
            submarine_x=next_submarine_x,
            submarine_active=next_submarine_active,
            submarine_timer=submarine_timer,
            wb_heli_x=next_wb_heli_x,
            wb_heli_active=next_wb_heli_active,
            wb_heli_timer=wb_heli_timer,
            wb_flyer_x=next_wb_flyer_x,
            wb_flyer_y=next_wb_flyer_y.astype(jnp.int32),
            wb_flyer_active=next_wb_flyer_active,
            wb_flyer_timer=wb_flyer_timer,
            sub_torp_x=next_torp_x,
            sub_torp_y=next_torp_y,
            sub_torp_active=next_torp_active,
            sub_fired=next_sub_fired,
            ## A fresh helicopter pass starts with a fresh bomb allowance.
            ## Needed because the daylight scene chains passes back to
            ## back, so the helicopter never goes inactive and the
            ## per-pass counter would otherwise stick at its maximum.
            helicopter_bombs_dropped=jnp.where(
                spawn_helicopter, jnp.array(0, dtype=jnp.int32), state.helicopter_bombs_dropped
            ),
            helicopter_bomb_timer=jnp.where(
                spawn_helicopter, jnp.array(0, dtype=jnp.int32), state.helicopter_bomb_timer
            ),
            ship_x=next_ship_x,
            ship_active=next_ship_active,
            ship_timer=ship_timer,
            base_x=next_base_x.astype(jnp.int32),
            base_active=next_base_active,
            goal_x=next_goal_x.astype(jnp.int32),
            goal_active=next_goal_active,
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
        ## In the daylight scene the bomb keeps sinking well under the
        ## surface (video: it fades around the submarine's depth)
        bomb_floor = jnp.where(
            state.stage == 3,
            jnp.array(self.consts.WC_BOMB_FLOOR, dtype=jnp.int32),
            jnp.array(ground, dtype=jnp.int32),
        )
        heli_bomb_active = jnp.logical_and(
            state.helicopter_bomb_active, heli_bomb_y < bomb_floor
        )

        laser_x = jnp.where(state.satellite_laser_active, state.satellite_laser_x, -1)
        laser_y = jnp.where(
            state.satellite_laser_active,
            state.satellite_laser_y + self.consts.SATELLITE_LASER_FALL_SPEED,
            -1,
        )
        ## On land the bolt vanishes at the ground line. In the water it
        ## keeps sinking below the surface and only detonates deeper down
        ## (measured: bolt visible under water to ~y134-137 before the
        ## explosion appears at the surface).
        laser_floor = jnp.where(
            state.stage == 1,
            jnp.array(self.consts.WATER_LASER_FLOOR, dtype=jnp.int32),
            jnp.array(ground, dtype=jnp.int32),
        )
        laser_active = jnp.logical_and(state.satellite_laser_active, laser_y < laser_floor)

        ## Splash detonation, measured frame-exact in ALE: the spent bolt
        ## becomes the green frogman at the impact column (narrow pose the
        ## first frame), riding the world scroll for exactly 120 frames.
        ## A scuba diver anywhere on screen suppresses this conversion: the
        ## bolt simply disappears at its normal water floor instead.
        detonate = jnp.logical_and(
            jnp.logical_and(state.satellite_laser_active, laser_y >= laser_floor),
            state.stage == 1,
        )
        spawn_splash = jnp.logical_and(
            detonate,
            jnp.logical_not(
                jnp.logical_or(state.scuba_active, state.splash_active)
            ),
        )
        refresh_splash = jnp.logical_and(
            jnp.logical_and(detonate, state.splash_active),
            jnp.logical_not(state.scuba_active),
        )
        splash_x = jnp.where(
            spawn_splash,
            laser_x, ## the frogman surfaces at [laser_x, laser_x+19], not centered
            state.splash_x,
        )
        ## If a diver enters while an older radioactive splash is alive, the
        ## diver rule wins immediately; both radioactive forms never coexist.
        splash_active = jnp.logical_and(
            jnp.logical_or(state.splash_active, spawn_splash),
            jnp.logical_not(state.scuba_active),
        )
        splash_age = jnp.where(
            state.scuba_active,
            0,
            jnp.where(
                jnp.logical_or(spawn_splash, refresh_splash),
                0,
                state.splash_age,
            ),
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
        ## The bomb and the satellite laser share one hardware sprite slot
        ## in the real game: they were never airborne together in 3k+
        ## measured frames, naturally or forced. Enforce the exclusivity.
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
        ## Direction picked once on release, but NOT aimed at the player:
        ## measured across every recorded drop, the bomb trails the flying
        ## helicopter 1px left per frame, and falls straight down only when
        ## the helicopter releases near the left edge. A rightward bomb was
        ## never observed, player position notwithstanding.
        release_vx = jnp.where(
            state.helicopter_x > 31,
            -self.consts.HELICOPTER_BOMB_SPEED_X,
            0,
        ).astype(jnp.int32)
        heli_bomb_vx = jnp.where(drop_bomb, release_vx, state.helicopter_bomb_vx)
        heli_bomb_active = jnp.logical_or(heli_bomb_active, drop_bomb)

        ## 3. Satellite laser. Two triggers, one per scene, both measured:
        ## - Land: a simple kitchen timer. Counts down while a satellite is
        ##   on screen, drops from its belly at zero, rewinds. Parked at
        ##   full while no satellite is around, so every new pass starts a
        ##   fresh countdown.
        ## - Water: the timer is ignored. The satellite releases the laser
        ##   the moment its belly passes over the player's column (the drop
        ##   column always matched the player column in ALE), a couple of
        ##   times per pass at most.
        laser_timer = jnp.where(
            state.satellite_active,
            jnp.maximum(state.satellite_laser_timer - 1, 0),
            jnp.array(self.consts.SATELLITE_LASER_DROP_PERIOD, dtype=jnp.int32),
        )
        timer_drop = jnp.logical_and(state.satellite_active, laser_timer == 0)

        ## Water rule, nailed with RAM-injection scans in ALE: at discrete
        ## check moments the satellite fires iff the drop column sits 1..95
        ## px AHEAD (right) of the player's hull -- never while still
        ## behind it. With a parked player this looks like "drops when
        ## overhead", but the window is wide open to the right.
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
            ## One laser at a time, and never while the helicopter bomb is
            ## airborne -- they share the single projectile slot.
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
        ## Count drops per pass, forget once the satellite is gone
        lasers_dropped = jnp.where(
            state.satellite_active,
            state.satellite_lasers_dropped + drop_laser.astype(jnp.int32),
            0,
        )

        return state.replace(
            satellite_lasers_dropped=lasers_dropped,
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
            splash_x=splash_x.astype(jnp.int32),
            splash_active=splash_active,
            splash_age=splash_age,
        )

    def _is_done(self, state: JamesBondState) -> chex.Array:
        return self._get_done(state)

    def _resolve_collisions(self, state: JamesBondState) -> JamesBondState:
        """Run all collision systems after movement and object updates."""

        state = self._resolve_player_bullet_collisions(state)
        state = self._resolve_player_wbullet_collisions(state)
        state = self._resolve_waterb_ball_shot(state)
        state = self._resolve_water_shots(state)
        state = self._resolve_bullet_player_collisions(state)
        state = self._resolve_pit_player_collisions(state)
        state = self._resolve_splash_player_collisions(state)
        state = self._resolve_waterb_collisions(state)
        state = self._resolve_waterc_contacts(state)
        state = self._resolve_oil_rig_collision(state)
        return state

    def _resolve_waterb_ball_shot(self, state: JamesBondState) -> JamesBondState:
        """Water B: the anti-air shot pops the pink ball for 500.

        The hit that completes WB_BALL_HITS_TO_EXIT starts the scene-clear
        freeze instead of removing the ball: it hangs in its hit pose for
        the whole colour cycle (video), and the freeze's end sweeps it.
        """

        ball_hit = jnp.logical_and(
            jnp.logical_and(state.stage == 2, state.win_timer == 0),
            jnp.logical_and(
                jnp.logical_and(state.player_bullet_active, state.wb_heli_active),
                _aabb_overlap(
                    state.player_bullet_x, state.player_bullet_y,
                    self.consts.BULLET_WIDTH, self.consts.BULLET_HEIGHT,
                    state.wb_heli_x, jnp.array(self.consts.WB_HELI_Y, dtype=jnp.int32),
                    self.consts.WB_HELI_WIDTH, self.consts.WB_HELI_HEIGHT,
                ),
            ),
        )
        hits = state.wb_ball_hits + ball_hit.astype(jnp.int32)
        final_hit = jnp.logical_and(ball_hit, hits >= self.consts.WB_BALL_HITS_TO_EXIT)
        bullet_keep = state.player_bullet_active & (~ball_hit)

        def park(keep, v):
            return jnp.where(keep, v, jnp.array(-1, dtype=v.dtype))

        return state.replace(
            score=(state.score + ball_hit.astype(jnp.int32) * self.consts.SCORE_BALL).astype(jnp.int32),
            wb_ball_hits=hits,
            wb_heli_active=state.wb_heli_active & ((~ball_hit) | final_hit),
            win_timer=jnp.where(
                final_hit,
                jnp.array(self.consts.WIN_ANIMATION_FRAMES, dtype=jnp.int32),
                state.win_timer,
            ),
            player_bullet_active=bullet_keep,
            player_bullet_step=park(bullet_keep, state.player_bullet_step),
            player_bullet_x=park(bullet_keep, state.player_bullet_x),
            player_bullet_y=park(bullet_keep, state.player_bullet_y),
        )

    def _resolve_water_shots(self, state: JamesBondState) -> JamesBondState:
        """Water B and C: the player's rounds finally hit something.

        Read off the longplay videos: the anti-air shot destroys a climbing
        rocket and the depth charge a submerged or surfacing one -- worth
        200 in water B (6500 -> 6700 the moment the shot touched it) and
        100 in the daylight scene; the depth charge also sinks the
        submarine for 200, and in the daylight scene the anti-air shot
        pops the falling debris for 100. The used round is consumed on
        impact.
        """

        in_water_b = state.stage == 2
        in_water_c = state.stage == 3
        in_scene = in_water_b | in_water_c
        rocket_w = jnp.where(in_water_c, self.consts.WC_ROCKET_WIDTH, self.consts.ROCKET_WIDTH)
        rocket_h = jnp.where(in_water_c, self.consts.WC_ROCKET_HEIGHT, self.consts.ROCKET_HEIGHT)

        def rocket_hits(bx, by, active):
            return jnp.logical_and(
                jnp.logical_and(active, state.rocket_active),
                _aabb_overlap(
                    bx, by,
                    self.consts.BULLET_WIDTH, self.consts.BULLET_HEIGHT,
                    state.rocket_x, state.rocket_y,
                    rocket_w, rocket_h,
                ),
            )

        air_hits = jnp.logical_and(
            in_scene,
            rocket_hits(state.player_bullet_x, state.player_bullet_y, state.player_bullet_active),
        )
        water_hits = jnp.logical_and(
            in_scene,
            rocket_hits(state.player_wbullet_x, state.player_wbullet_y, state.player_wbullet_active),
        )
        rocket_hit = air_hits | water_hits
        rocket_value = jnp.where(in_water_c, self.consts.SCORE_ROCKET_SHOT, self.consts.SCORE_ROCKET)
        ## The anti-air shot also pops the falling / floating debris
        debris_hits = jnp.logical_and(
            jnp.logical_and(in_scene, state.player_bullet_active),
            jnp.logical_and(
                state.wb_flyer_active,
                _aabb_overlap(
                    state.player_bullet_x, state.player_bullet_y,
                    self.consts.BULLET_WIDTH, self.consts.BULLET_HEIGHT,
                    state.wb_flyer_x, state.wb_flyer_y,
                    self.consts.WB_FLYER_WIDTH, self.consts.WB_FLYER_HEIGHT,
                ),
            ),
        )
        sub_hit = jnp.logical_and(
            jnp.logical_and(in_scene, state.player_wbullet_active),
            jnp.logical_and(
                state.submarine_active,
                _aabb_overlap(
                    state.player_wbullet_x, state.player_wbullet_y,
                    self.consts.BULLET_WIDTH, self.consts.BULLET_HEIGHT,
                    state.submarine_x, jnp.array(self.consts.SUBMARINE_Y, dtype=jnp.int32),
                    self.consts.SUBMARINE_WIDTH, self.consts.SUBMARINE_HEIGHT,
                ),
            ),
        )
        air_used = jnp.any(air_hits) | jnp.any(debris_hits)
        water_used = jnp.any(water_hits) | sub_hit
        gained = (
            jnp.sum(rocket_hit.astype(jnp.int32)) * rocket_value
            + jnp.sum(debris_hits.astype(jnp.int32)) * self.consts.SCORE_DEBRIS_SHOT
            + sub_hit.astype(jnp.int32) * self.consts.SCORE_SUBMARINE_SHOT
        )

        def park(keep, v):
            return jnp.where(keep, v, jnp.array(-1, dtype=v.dtype))

        air_keep = state.player_bullet_active & (~air_used)
        water_keep = state.player_wbullet_active & (~water_used)
        return state.replace(
            score=(state.score + gained).astype(jnp.int32),
            rocket_active=state.rocket_active & (~rocket_hit),
            wb_flyer_active=state.wb_flyer_active & (~debris_hits),
            submarine_active=state.submarine_active & (~sub_hit),
            player_bullet_active=air_keep,
            player_bullet_step=park(air_keep, state.player_bullet_step),
            player_bullet_x=park(air_keep, state.player_bullet_x),
            player_bullet_y=park(air_keep, state.player_bullet_y),
            player_wbullet_active=water_keep,
            player_wbullet_step=park(water_keep, state.player_wbullet_step),
            player_wbullet_x=park(water_keep, state.player_wbullet_x),
            player_wbullet_y=park(water_keep, state.player_wbullet_y),
        )

    def _resolve_waterc_contacts(self, state: JamesBondState) -> JamesBondState:
        """Daylight scene contacts the other scenes do not have.

        The falling / floating red debris and the steamship's deck and
        hull cost a life; touching the green objective completes the
        mission instead (the freeze and bonus run in the step loop).
        A win outranks a death in the same frame.
        """

        in_water_c = state.stage == 3

        def touch(ox, oy, ow, oh):
            return _aabb_overlap(
                state.player_x, state.player_y,
                self.consts.PLAYER_COLLISION_WIDTH, self.consts.PLAYER_COLLISION_HEIGHT,
                ox, oy, ow, oh,
            )

        debris_hit = jnp.logical_and(
            jnp.logical_or(state.stage == 2, in_water_c),
            jnp.any(jnp.logical_and(
                state.wb_flyer_active,
                touch(state.wb_flyer_x, state.wb_flyer_y,
                      self.consts.WB_FLYER_WIDTH, self.consts.WB_FLYER_HEIGHT),
            )),
        )
        ship_hit = jnp.logical_and(
            jnp.logical_and(in_water_c, state.ship_active),
            touch(
                state.ship_x, jnp.array(self.consts.WC_SHIP_HULL_TOP, dtype=jnp.int32),
                self.consts.WC_SHIP_WIDTH,
                self.consts.WC_SHIP_Y + self.consts.WC_SHIP_HEIGHT - self.consts.WC_SHIP_HULL_TOP,
            ),
        )
        goal_hit = jnp.logical_and(
            jnp.logical_and(in_water_c, state.goal_active),
            touch(state.goal_x, jnp.array(self.consts.WC_GOAL_Y, dtype=jnp.int32),
                  self.consts.WC_GOAL_WIDTH, self.consts.WC_GOAL_HEIGHT),
        )
        lethal = jnp.logical_and(
            debris_hit | ship_hit,
            jnp.logical_not(goal_hit),
        )
        can_take_damage = state.hit_cooldown <= 0
        took_damage = jnp.logical_and(lethal, can_take_damage)
        win = jnp.logical_and(goal_hit, state.win_timer == 0)

        return state.replace(
            goal_active=jnp.logical_and(state.goal_active, jnp.logical_not(goal_hit)),
            win_timer=jnp.where(
                win,
                jnp.array(self.consts.WIN_ANIMATION_FRAMES, dtype=jnp.int32),
                state.win_timer,
            ),
            lives=jnp.maximum(
                0, state.lives - took_damage.astype(jnp.int32)
            ).astype(jnp.int32),
            hit_cooldown=jnp.where(
                took_damage,
                jnp.array(self.consts.HIT_COOLDOWN_STEPS, dtype=jnp.int32),
                state.hit_cooldown,
            ),
            death_timer=jnp.where(
                took_damage,
                jnp.array(self.consts.DEATH_ANIMATION_FRAMES, dtype=jnp.int32),
                state.death_timer,
            ),
        )

    def _resolve_oil_rig_collision(self, state: JamesBondState) -> JamesBondState:
        """Side contact with the oil rig is lethal; landing on top is handled
        in _update_stage (it advances the scene, so it must NOT also kill).
        """
        overlap = jnp.logical_and(
            state.oil_rig_active,
            _aabb_overlap(
                state.player_x, state.player_y,
                self.consts.PLAYER_COLLISION_WIDTH, self.consts.PLAYER_COLLISION_HEIGHT,
                state.oil_rig_x, state.oil_rig_y,
                self.consts.OIL_RIG_WIDTH, self.consts.OIL_RIG_HEIGHT,
            ),
        )
        on_top = jnp.logical_and(
            jnp.logical_and(
                state.player_x + self.consts.PLAYER_COLLISION_WIDTH > state.oil_rig_x,
                state.player_x < state.oil_rig_x + self.consts.OIL_RIG_WIDTH,
            ),
            jnp.logical_and(
                state.player_y + self.consts.PLAYER_COLLISION_HEIGHT >= state.oil_rig_y,
                state.player_y <= state.oil_rig_y + self.consts.OIL_RIG_TOP_LAND_MARGIN,
            ),
        )
        side_hit = jnp.logical_and(overlap, jnp.logical_not(on_top))
        can_take_damage = state.hit_cooldown <= 0
        took_damage = jnp.logical_and(side_hit, can_take_damage)
        return state.replace(
            lives=jnp.maximum(
                0, state.lives - took_damage.astype(jnp.int32)
            ).astype(jnp.int32),
            hit_cooldown=jnp.where(
                took_damage,
                jnp.array(self.consts.HIT_COOLDOWN_STEPS, dtype=jnp.int32),
                state.hit_cooldown,
            ),
            death_timer=jnp.where(
                took_damage,
                jnp.array(self.consts.DEATH_ANIMATION_FRAMES, dtype=jnp.int32),
                state.death_timer,
            ),
        )

    def _resolve_waterb_collisions(self, state: JamesBondState) -> JamesBondState:
        """Second water scene contacts.

        Ramming the floating rocket is the scene's only score (+200), and
        in the real game the ram usually costs a life as well -- both
        effects fire here, the damage under the usual cooldown. The
        submarine and the two flyers just hurt: the submarine can only
        reach a diving boat, the flyers only a jumping one.
        """

        def touch(ox, oy, ow, oh):
            return _aabb_overlap(
                state.player_x,
                state.player_y,
                self.consts.PLAYER_COLLISION_WIDTH,
                self.consts.PLAYER_COLLISION_HEIGHT,
                ox, oy, ow, oh,
            )

        ## One entry per rocket slot; ramming pays only in water B (the
        ## daylight scene scores through the player's rounds instead)
        in_water_c = state.stage == 3
        rocket_hit_slots = jnp.logical_and(
            state.rocket_active,
            touch(state.rocket_x, state.rocket_y,
                  jnp.where(in_water_c, self.consts.WC_ROCKET_WIDTH, self.consts.ROCKET_WIDTH),
                  jnp.where(in_water_c, self.consts.WC_ROCKET_HEIGHT, self.consts.ROCKET_HEIGHT)),
        )
        rocket_hit = jnp.any(rocket_hit_slots)
        rocket_score = jnp.logical_and(rocket_hit, state.stage == 2)
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
        ## The submarine's double-dot shot, on its diagonal or its run
        ## along the surface, costs a life too
        shot_hit = jnp.logical_and(
            state.sub_torp_active,
            touch(state.sub_torp_x, state.sub_torp_y,
                  self.consts.SUB_SHOT_WIDTH, self.consts.SUB_SHOT_HEIGHT),
        )
        ## The rocket debris (falling or floating) is handled together
        ## with the other daylight-style contacts in _resolve_waterc_contacts.

        any_hit = rocket_hit | submarine_hit | heli_hit | shot_hit
        can_take_damage = state.hit_cooldown <= 0
        took_damage = jnp.logical_and(any_hit, can_take_damage)

        return state.replace(
            score=state.score + rocket_score.astype(jnp.int32) * self.consts.SCORE_ROCKET,
            rocket_active=jnp.logical_and(state.rocket_active, ~rocket_hit_slots),
            sub_torp_active=jnp.logical_and(state.sub_torp_active, ~shot_hit),
            lives=jnp.maximum(
                0, state.lives - took_damage.astype(jnp.int32)
            ).astype(jnp.int32),
            hit_cooldown=jnp.where(
                took_damage,
                jnp.array(self.consts.HIT_COOLDOWN_STEPS, dtype=jnp.int32),
                state.hit_cooldown,
            ),
            death_timer=jnp.where(
                took_damage,
                jnp.array(self.consts.DEATH_ANIMATION_FRAMES, dtype=jnp.int32),
                state.death_timer,
            ),
        )

    def _resolve_splash_player_collisions(self, state: JamesBondState) -> JamesBondState: ## TODO: Correct?
        """One life of damage from the radioactive splash hazard.

        The laser splash explosion straddles the surface and kills on
        near-contact (measured: within ~1px, submerged or afloat); only
        a clearly airborne boat passes over it safely.
        """

        ## Splash: the measured kill window is boat_x in
        ## [splash_x - 9, splash_x + 19], plus an altitude gate
        x_touch = jnp.logical_and(
            state.player_x >= state.splash_x - 9,
            state.player_x <= state.splash_x + 19,
        )
        low_enough = state.player_y >= self.consts.SPLASH_SAFE_PLAYER_Y
        splash_hit = jnp.logical_and(
            jnp.logical_and(
                state.splash_active,
                jnp.logical_not(state.scuba_active),
            ),
            jnp.logical_and(x_touch, low_enough),
        )

        can_take_damage = state.hit_cooldown <= 0

        return state.replace(
            lives=jnp.maximum(
                0, state.lives - splash_hit.astype(jnp.int32)
            ).astype(jnp.int32),
            hit_cooldown=jnp.where(
                splash_hit,
                jnp.array(self.consts.HIT_COOLDOWN_STEPS, dtype=jnp.int32),
                state.hit_cooldown,
            ),
            death_timer=jnp.where(
                splash_hit,
                jnp.array(self.consts.DEATH_ANIMATION_FRAMES, dtype=jnp.int32),
                state.death_timer,
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
            death_timer=jnp.where(
                took_damage,
                jnp.array(self.consts.DEATH_ANIMATION_FRAMES, dtype=jnp.int32),
                state.death_timer,
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
            death_timer=jnp.where(
                took_damage,
                jnp.array(self.consts.DEATH_ANIMATION_FRAMES, dtype=jnp.int32),
                state.death_timer,
            ),
        )

    def collectible_collisions_logic(self, state: JamesBondState) -> JamesBondState:
        """Collect active diamonds that overlap a player shot.

        Both shots count: the land round and the water anti-air round fly
        the same up-forward path, and shooting the floating gem is worth
        +50 in every scene (verified in ALE on land and over the water).
        """

        overlap = _aabb_overlap(
            state.player_bullet_x,
            state.player_bullet_y,
            self.consts.BULLET_WIDTH,
            self.consts.BULLET_HEIGHT,
            state.diamond_x,
            state.diamond_y,
            self.consts.DIAMOND_COLLISION_WIDTH,
            self.consts.DIAMOND_COLLISION_HEIGHT,
        )

        collected = jnp.logical_and(
            jnp.logical_and(state.diamond_active, state.player_bullet_active),
            overlap,
        )

        player_bullet_active = jnp.logical_and(
            state.player_bullet_active, ~collected
        )

        new_score = state.score + collected * self.consts.SCORE_DIAMOND

        def park(active, v):
            return jnp.where(active, v, -1)

        return state.replace(
            diamond_shot=collected,
            diamond_active = jnp.logical_and(
                state.diamond_active, ~collected
            ),
            player_bullet_active=player_bullet_active,
            player_bullet_step=park(player_bullet_active, state.player_bullet_step),
            player_bullet_x=park(player_bullet_active, state.player_bullet_x),
            player_bullet_y=park(player_bullet_active, state.player_bullet_y),
            score=new_score
        )

    def scuba_collisions_logic(self, state: JamesBondState) -> JamesBondState:

        overlap = _aabb_overlap(
            state.player_wbullet_x,
            state.player_wbullet_y,
            self.consts.BULLET_WIDTH,
            self.consts.BULLET_HEIGHT,
            state.scuba_x,
            state.scuba_y,
            self.consts.SCUBA_WIDTH,
            self.consts.SCUBA_HEIGHT,
        )

        hit = jnp.logical_and(
            jnp.logical_and(state.scuba_active, state.player_wbullet_active),
            overlap,
        )

        player_wbullet_active = jnp.logical_and(
            state.player_wbullet_active, ~hit
        )

        new_score = state.score + hit * self.consts.SCORE_SCUBA

        def park(active, v):
            return jnp.where(active, v, -1)

        return state.replace(
            scuba_active = jnp.logical_and(
                state.scuba_active, ~hit
            ),
            player_wbullet_active=player_wbullet_active,
            player_wbullet_step=park(player_wbullet_active, state.player_wbullet_step),
            player_wbullet_x=park(player_wbullet_active, state.player_wbullet_x),
            player_wbullet_y=park(player_wbullet_active, state.player_wbullet_y),
            score=new_score
        )
    
    def _resolve_player_bullet_collisions(self, state: JamesBondState) -> JamesBondState:
        return lax.cond(
            state.player_bullet_active,
            self.collectible_collisions_logic,
            lambda s: s,
            state
        )

    def _resolve_player_wbullet_collisions(self, state: JamesBondState) -> JamesBondState:
        return lax.cond(
            state.player_wbullet_active,
            self.scuba_collisions_logic,
            lambda s: s,
            state
        )

    def _get_reward( ## TODO: Wrong logic
        self, previous_state: JamesBondState, state: JamesBondState
    ) -> chex.Array:
        """Calculate reward from collision-driven state transitions."""

        return state.score - previous_state.score

        """
        score_gained = jnp.maximum(state.score - previous_state.score, 0)
            diamonds_collected = jnp.where(
                state.collected_diamond, ## TODO: REMOVE
                jnp.floor_divide(score_gained, self.consts.SCORE_DIAMOND),
                0,
            )
            lives_lost = jnp.maximum(previous_state.lives - state.lives, 0)
    
            return (
                jnp.array(self.consts.REWARD_STEP, dtype=jnp.float32)
                + diamonds_collected.astype(jnp.float32) * self.consts.REWARD_DIAMOND
                + lives_lost.astype(jnp.float32) * self.consts.REWARD_LOST_LIFE
            )
        """

    def _get_done(self, state: JamesBondState) -> chex.Array:
        ## Stage 4 is not a scene: the mission-complete freeze sets it
        ## once the bonus has paid out, so the episode ends there.
        return jnp.logical_or(
            state.lives <= 0,
            state.stage >= 4,
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

        ## The first two water scenes have a gray sky covering the measured
        ## rows 29-120, meeting the water at 121; it goes under the stars
        ## so the stars still twinkle on it like in the real scenes. The
        ## daylight scene paints the same band blue, with clouds and no stars.
        raster = jax.lax.cond(
            jnp.logical_or(state.stage == 1, state.stage == 2),
            lambda r: self.jr.render_at_clipped(r, 4, 29, self.SHAPE_MASKS['water_sky']),
            lambda r: r,
            raster,
        )
        raster = jax.lax.cond(
            state.stage >= 3,
            lambda r: self._render_clouds(
                self.jr.render_at_clipped(r, 4, 29, self.SHAPE_MASKS['wc_sky']), state
            ),
            lambda r: r,
            raster,
        )
        ## Measured: on the very first frame of a water-scene death the
        ## whole sky flashes #6f6f6f (the death_timer sits at its full
        ## value for exactly that one frame)
        raster = jax.lax.cond(
            jnp.logical_and(
                state.stage >= 1,
                state.death_timer >= self.consts.DEATH_ANIMATION_FRAMES - 2,
            ),
            lambda r: self.jr.render_at_clipped(r, 4, 29, self.SHAPE_MASKS['death_flash']),
            lambda r: r,
            raster,
        )
        ## Water B's scene-clear freeze: the sky flickers between the two
        ## grays for its first WB_EXIT_FLICKER_FRAMES frames (video: an
        ## irregular dark/light strobe, then the plain sky while the boat
        ## keeps colour-cycling)
        flicker = jnp.logical_and(
            state.stage == 2,
            state.win_timer > self.consts.WIN_ANIMATION_FRAMES - self.consts.WB_EXIT_FLICKER_FRAMES,
        )
        flicker_sprite = jnp.where(state.win_timer % 3 == 0, 0, 1)
        raster = jax.lax.cond(
            jnp.logical_and(flicker, state.win_timer % 3 != 2),
            lambda r: jax.lax.switch(
                flicker_sprite,
                [
                    lambda rr: self.jr.render_at_clipped(rr, 4, 29, self.SHAPE_MASKS['sky_flash']),
                    lambda rr: self.jr.render_at_clipped(rr, 4, 29, self.SHAPE_MASKS['death_flash']),
                ],
                r,
            ),
            lambda r: r,
            raster,
        )
        ## Bright full-screen flash for the first few frames as the oil rig
        ## arrives -- a quick strike, while the rig itself lingers after.
        ## Flash fires at BOTH appearances: the first few frames of the RIGHT
        ## phase and the first few frames of the LEFT phase, keyed to oil_rig_seq.
        _strike = self.consts.OIL_RIG_STRIKE_LEN
        flash_right = jnp.logical_and(
            state.oil_rig_seq <= self.consts.OIL_RIG_SEQ_TOTAL,
            state.oil_rig_seq > self.consts.OIL_RIG_SEQ_TOTAL - _strike,
        )
        flash_left = jnp.logical_and(
            state.oil_rig_seq <= self.consts.OIL_RIG_SEQ_LEFT_START,
            state.oil_rig_seq > self.consts.OIL_RIG_SEQ_LEFT_START - _strike,
        )
        rig_flash = jnp.logical_and(
            state.stage == 1,
            jnp.logical_or(flash_right, flash_left),
        )
        def _rig_flash(r):
            pos = jnp.array([[0, 29]], dtype=jnp.int32)   # sky band only, starts below the HUD
            size = jnp.array([[self.consts.SCREEN_WIDTH, 91]], dtype=jnp.int32)  # rows 29..120
            return self.jr.draw_rects(r, pos, size, 13)  # id 13 = white (236,236,236) - visible flash over grey sky
        raster = jax.lax.cond(rig_flash, _rig_flash, lambda r: r, raster)
        raster = jax.lax.cond(
            state.stage < 3,
            lambda r: self._render_stars(r, state),
            lambda r: r,
            raster,
        )
        ## Terrain follows the scene: dry land in stage 0, water in stage 1
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
        raster = self._render_helicopter_melee(raster, state)
        raster = self._render_satellite(raster, state)
        raster = self._render_scuba(raster, state)
        raster = self._render_splash(raster, state)
        raster = self._render_waterb(raster, state)
        raster = self._render_oil_rig(raster, state)

        raster = self._render_bullets(raster, state)
        raster = self._render_sinking_bolt(raster, state)

        ## Render life counter: the HUD shows RESERVE lives (max 3 icons),
        ## not the life currently being played, matching the real game
        raster = self.jr.render_indicator(
            raster, 9, 184,
            jnp.maximum(state.lives - 1, 0),
            self.SHAPE_MASKS['life'], 16, 3,
        )

        ## Render black borders on the sides of the screen; TODO: Optimize
        raster = self.jr.render_at(
            raster,
            0,
            2,
            self.SHAPE_MASKS['black_border']
        )
        raster = self.jr.render_at(
            raster,
            210,
            2,
            self.SHAPE_MASKS['black_border']
        )
        
        ## Render Score counter
        score_digits = self.jr.int_to_digits(state.score, 4) ## TODO: Max score 4 digits?
        raster = self.jr.render_label(raster, 95, 15, score_digits, self.SHAPE_MASKS['score_digits'], 8, 4) ## TODO: Position offset per digit?

        return self.jr.render_from_palette(raster, self.PALETTE)

    def _render_ground(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:
        """Draw the static ground strip using the existing ground sprite.

        The sprite's first row is the gray road top, which sits at row 124
        in the real game -- BELOW the player, whose hull rides rows
        119-122 with the wheels touching the road. Drawing it at 119 put
        the road at roof height and sank the car into the ground.
        """

        return self.jr.render_at_clipped(
            raster,
            4,    # x - real playfield left edge (columns 0-7 stay black like ALE)
            123,  # y - road top just under the wheels, like the real rows
            self.SHAPE_MASKS['ground'],
        )

    def _render_water(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:
        """Draw the water scene terrain: the water body and the seabed.

        Both sprites were extracted from the real water scene. The water
        band starts at the same row the land ground uses, so the boat rides
        the surface exactly where the car drives; the seabed silhouette
        sits at the bottom of the water body.
        """

        ## The second water scene runs the same layout with darker water,
        ## the daylight scene with navy water. The water surface is row
        ## 121 in the real game: the boat's hull (119-122) rides half
        ## above, half below the waterline.
        water_idx = jnp.clip(state.stage - 1, 0, 2)
        raster = jax.lax.switch(
            water_idx,
            [
                lambda r: self.jr.render_at_clipped(r, 4, 121, self.SHAPE_MASKS['water']),
                lambda r: self.jr.render_at_clipped(r, 4, 121, self.SHAPE_MASKS['water_b']),
                lambda r: self.jr.render_at_clipped(r, 4, 121, self.SHAPE_MASKS['wc_water']),
            ],
            raster,
        )
        ## The seabed is a 160px repeating strip that scrolls left with the
        ## world, 1px every 4th frame. Drawing the pattern twice, one strip
        ## width apart, keeps the wrap seamless. The daylight scene uses
        ## the green recolor, and its hills stop once the sunken base
        ## scrolls in (video: flat seabed under the base).
        scroll = (state.step_count // 4) % 160

        def draw_seabed(r, mask_name):
            r = self.jr.render_at_clipped(
                r,
                4 - scroll,
                158,  # y - measured seabed top row
                self.SHAPE_MASKS[mask_name],
            )
            return self.jr.render_at_clipped(
                r,
                4 - scroll + 160,
                158,
                self.SHAPE_MASKS[mask_name],
            )

        return jax.lax.cond(
            state.stage >= 3,
            lambda r: jax.lax.cond(
                state.base_active,
                lambda rr: rr,
                lambda rr: draw_seabed(rr, 'wc_seabed'),
                r,
            ),
            lambda r: draw_seabed(r, 'seabed'),
            raster,
        )

    def _render_clouds(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:
        """Daylight sky: two clouds per 160px strip, a low one and, 62px
        to its right, a high one, drifting left at 0.5 px/frame -- twice
        the seabed scroll (video parallax). Drawn twice for a seamless wrap.
        """

        scroll = (state.step_count // 2) % 160
        for strip in (0, 160):
            raster = self.jr.render_at_clipped(
                raster, 4 - scroll + strip, 43, self.SHAPE_MASKS["wc_cloud"])
            raster = self.jr.render_at_clipped(
                raster, 4 - scroll + strip + 62, 30, self.SHAPE_MASKS["wc_cloud"])
        return raster

    def _render_scuba(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:
        """Draw normal scuba or the satellite-drop radioactive figure.

        Normal scuba switches swim frames every 15 frames. When proximity
        activates radioactivity, replace the swimmer with the exact same
        narrow/wide two-pose figure used by a radioactive satellite drop.
        """

        normal_idx = (state.step_count // 15) % 2
        radioactive_narrow = ((state.scuba_radioactive_age + 6) // 7) % 2 == 0

        def draw_normal(r):
            return self.jr.render_at_clipped(
                r,
                state.scuba_x,
                state.scuba_y,
                self.SHAPE_MASKS['scuba'][normal_idx],
            )

        def draw_radioactive(r):
            return jax.lax.cond(
                radioactive_narrow,
                lambda rr: self.jr.render_at_clipped(
                    rr,
                    state.scuba_x,
                    state.scuba_y,
                    self.SHAPE_MASKS['splash'],
                ),
                lambda rr: self.jr.render_at_clipped(
                    rr,
                    state.scuba_x - 4,
                    state.scuba_y,
                    self.SHAPE_MASKS['splash_wide'],
                ),
                r,
            )

        def draw_active(r):
            return jax.lax.cond(
                state.scuba_radioactive,
                draw_radioactive,
                draw_normal,
                r,
            )

        return jax.lax.cond(state.scuba_active, draw_active, lambda r: r, raster)

    def _render_splash(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:
        """Draw the splash only when no scuba diver is present.

        Frame-exact from ALE: the narrow pose shows for exactly 1 frame at
        spawn, then strict 7-frame phases alternate starting with the wide
        pose (which reaches 4px further left and carries the yellow
        under-glow rows). Green is visible every single frame -- only the
        under-glow toggles with the wide phase.
        """

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

        visible = jnp.logical_and(
            state.splash_active,
            jnp.logical_not(state.scuba_active),
        )
        return jax.lax.cond(visible, draw_fn, lambda r: r, raster)

    ##def _render_oil_rig(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:
    ##    """ Draw the oil rig only during active flase frames. """
    ##    is_visible = state.oil_rig_active & (state.oil_rig_visible_timer > 0)
    ##
    ##    draw_fn = lambda r: self.jr.render_at_clipped(
    ##        r,
    ##        state.oil_rig_x,
    ##        state.oil_rig_y,
    ##        self.SHAPE_MASKS['oil_rig'][0],
    ##    )
    ##
    ##    return jax.law.cond(is_visible, draw_fn, lambda r: r, raster)

    def _render_sinking_bolt(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:
        """A bolt landing while the frogman lives sinks radioactive-green.

        Measured: no second frogman spawns; the bar keeps falling under the
        waterline drawn in the frogman's green instead of yellow.
        """

        submerged = jnp.logical_and(
            jnp.logical_and(
                jnp.logical_not(state.scuba_active),
                jnp.logical_and(state.stage == 1, state.splash_active),
            ),
            jnp.logical_and(
                state.satellite_laser_active,
                state.satellite_laser_y > 119,
            ),
        )

        def draw_fn(r):
            return self.jr.render_at_clipped(
                r,
                state.satellite_laser_x,
                state.satellite_laser_y,
                self.SHAPE_MASKS['laser_green'],
            )

        return jax.lax.cond(submerged, draw_fn, lambda r: r, raster)

    def _render_oil_rig(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:
        """Draw the oil rig sprite while it is active (glimpsed in the flash)."""
        def draw_fn(r):
            return self.jr.render_at_clipped(
                r, state.oil_rig_x, state.oil_rig_y, self.SHAPE_MASKS['oil_rig'][0]
            )
        return jax.lax.cond(state.oil_rig_visible, draw_fn, lambda r: r, raster)

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
        ## (water B only; no flash was visible in the daylight footage).
        ## Keyed to the debris height: the bars fall 1px/frame from the
        ## burst row, and their clock is held while they fall.
        raster = one(
            raster,
            jnp.logical_and(
                state.stage == 2,
                jnp.any(jnp.logical_and(
                    state.wb_flyer_active,
                    state.wb_flyer_y < self.consts.WB_FLYER_Y + self.consts.SKY_FLASH_FRAMES,
                )),
            ),
            jnp.array(4, dtype=jnp.int32), jnp.array(29, dtype=jnp.int32), 'sky_flash',
        )
        ## Daylight ending: the sunken base and the objective above it
        raster = one(raster, state.base_active, state.base_x,
                     jnp.array(self.consts.WC_BASE_Y, dtype=jnp.int32), 'wc_base')
        raster = one(raster, state.goal_active, state.goal_x,
                     jnp.array(self.consts.WC_GOAL_Y, dtype=jnp.int32), 'wc_goal')
        ## Rocket pads and their debris, one draw per slot. Water B keeps
        ## its ALE-extracted 8x11 rocket; the daylight scene draws the
        ## small pyramid, with its flame while it climbs.
        in_water_c = state.stage == 3
        rocket_flying = state.rocket_age >= state.rocket_launch_age
        ## Daylight debris that reached the waterline is drawn as the
        ## red / pink sparkle, its two dot patterns swapping every few
        ## frames (video); everywhere else the two red bars are drawn.
        splash_pose = (state.step_count // self.consts.WC_SPLASH_FLIP_FRAMES) % 2
        debris_resting = state.wb_flyer_y >= self.consts.WC_DEBRIS_REST_Y
        for slot in range(self.consts.WC_ROCKET_SLOTS):
            raster = one(raster, state.rocket_active[slot] & (~in_water_c),
                         state.rocket_x[slot], state.rocket_y[slot], 'rocket')
            raster = one(raster, state.rocket_active[slot] & in_water_c & (~rocket_flying[slot]),
                         state.rocket_x[slot], state.rocket_y[slot], 'wc_rocket')
            raster = one(raster, state.rocket_active[slot] & in_water_c & rocket_flying[slot],
                         state.rocket_x[slot], state.rocket_y[slot], 'wc_rocket_fire')
            raster = one(raster, state.wb_flyer_active[slot] & (~debris_resting[slot]),
                         state.wb_flyer_x[slot], state.wb_flyer_y[slot], 'flyer_red')
            raster = jax.lax.cond(
                state.wb_flyer_active[slot] & debris_resting[slot],
                lambda r, s=slot: self.jr.render_at_clipped(
                    r, state.wb_flyer_x[s] - 1,
                    jnp.array(self.consts.WC_SPLASH_Y, dtype=jnp.int32),
                    self.SHAPE_MASKS['wc_splash'][splash_pose]),
                lambda r: r,
                raster,
            )
        raster = one(raster, state.submarine_active, state.submarine_x,
                     jnp.array(self.consts.SUBMARINE_Y, dtype=jnp.int32), 'submarine')
        ## Pink ball: solid / striped poses alternate in flight; during
        ## the scene-clear freeze it hangs in the striped (hit) pose
        ball_pose = jnp.where(
            state.win_timer > 0,
            1,
            (state.step_count // 8) % 2,
        )
        raster = jax.lax.cond(
            state.wb_heli_active,
            lambda r: self.jr.render_at_clipped(
                r, state.wb_heli_x, jnp.array(self.consts.WB_HELI_Y, dtype=jnp.int32),
                self.SHAPE_MASKS['wb_ball'][ball_pose]),
            lambda r: r,
            raster,
        )
        raster = one(raster, state.ship_active, state.ship_x,
                     jnp.array(self.consts.WC_SHIP_Y, dtype=jnp.int32), 'wc_ship')
        ## Submarine shot: the two stacked yellow dots
        raster = one(raster, state.sub_torp_active, state.sub_torp_x, state.sub_torp_y, 'sub_shot')

        return raster

    def _render_stars(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:
        """Scattered star field across the whole sky; two frames alternate
        slowly for a gentle twinkle."""
        idx = jnp.where((state.step_count // 24) % 2 == 0, 0, 1)
        return self.jr.render_at_clipped(raster, 4, 34, self.SHAPE_MASKS['stars'][idx])

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
        """Draw the player, color-cycling through the recolors while dying.

        The world clock freezes during the 59-frame death animation, so
        the cycle is driven by death_timer (which keeps counting down);
        the multiplier makes the recolor order look random like the real
        sprite's per-frame color roll.
        """

        ## The mission-complete freeze at the end of the daylight scene
        ## runs the very same colour cycle (video), just without a death.
        cycle_timer = jnp.maximum(state.death_timer, state.win_timer)
        sprite_idx = jnp.where(
            cycle_timer > 0,
            1 + (cycle_timer * 5) % 3,
            jnp.where(
                state.hit_cooldown > 0,
                jnp.where(state.step_count % 2 == 0, 1, 2),
                0
            )
        )

        return self.jr.render_at_clipped(
            raster,
            state.player_x,
            state.player_y,
            self.SHAPE_MASKS['car'][sprite_idx]
        )
    
    def _render_diamond(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:

        ## Sparkle alternation is a few frames per phase in the real game;
        ## flipping every single frame made the whole gem vibrate.
        sprite_idx = jnp.where(
            (state.step_count // 4) % 2 == 0,
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

        ## The flame blinks in irregular multi-frame bursts in the real
        ## game; a few frames per phase reads right without the strobe.
        sprite_idx = jnp.where(
            (state.step_count // 3) % 2 == 0,
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

        ## Rotor frames alternate every 2-3 frames in the real game
        sprite_idx = jnp.where(
            (state.step_count // 2) % 2 == 0,
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

        step = jnp.clip(
            state.helicopter_melee_step,
            0,
            self.consts.HELICOPTER_MELEE_SPRITE_STEPS.shape[0] - 1,
        )
        melee_idx = self.consts.HELICOPTER_MELEE_SPRITE_STEPS[step][0]

        draw_fn = lambda r: self.jr.render_at_clipped(
            r,
            state.helicopter_x + self.consts.HELICOPTER_MELEE_SPRITE_STEPS[step][1],
            state.helicopter_y + 7,
            self.SHAPE_MASKS['helicopter_melee'][jnp.maximum(melee_idx, 0)]
        )

        visible = jnp.logical_and(state.helicopter_active, melee_idx != -1)
        return jax.lax.cond(visible, draw_fn, lambda r: r, raster)

    def _render_satellite(self, raster: jnp.ndarray, state: JamesBondState) -> jnp.ndarray:

        draw_fn = lambda r: self.jr.render_at_clipped(
            r,
            state.satellite_x,
            state.satellite_y,
            self.SHAPE_MASKS['satellite'],
        )

        return jax.lax.cond(state.satellite_active, draw_fn, lambda r: r, raster)
    
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

            sprite_idx = jnp.where( ## Different sprite for water bullet
                i == 1,
                1,
                0
            )

            draw_fn = lambda r: self.jr.render_at_clipped(
                r,
                bullet_positions[i][0],
                bullet_positions[i][1],
                self.SHAPE_MASKS['bullet'][sprite_idx],
            )

            return jax.lax.cond(should_draw, draw_fn, lambda r: r, current_raster)

        return jax.lax.fori_loop(0, jnp.size(active_bullets), render_single_bullet, raster)
