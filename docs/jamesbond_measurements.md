# James Bond 007 — ALE measurement notes

Everything below was measured against the real ROM in ALE
(`ALE/Jamesbond-v5`, `frameskip=1`, `repeat_action_probability=0`, seed 0,
one `step()` = one console frame), during the full-game implementation on
the `test/jamesbond-full-game` branch. Coordinates are ALE screen pixels
(160 wide, 210 tall) and map 1:1 onto the JAX engine's coordinates.

## Scene structure

| scene | length (clean) | ends with |
|---|---|---|
| Land ("Diamonds Are Forever") | ~4454 frames | terrain turns to water |
| Water A | ~4435 frames | a dock scrolls in; boarding it pays +5000 |
| Water B (darker water) | never ended in a 22k-frame probe | treated as endless |

- Death animation: the whole scene freezes for exactly **59 frames** while the
  player sprite color-cycles; the life is deducted at the end; every actor and
  projectile despawns; the pit resets to x=124; the player respawns at x=29.
- Starting lives: **6**. Game over shows the Bond figure and "LEVEL: NOVICE".
- Mode 1 is an enemy-behavior variation of the same mission, not a new scene.

## Scoring (complete list found)

| points | cause |
|---|---|
| +50 | shooting the floating diamond (any scene; it cannot be touched — jump apex is ~30px short) |
| +5000 | boarding the dock at the end of Water A |
| +200 | ramming the floating rocket in Water B (usually costs a life too) |

Nothing else ever scored: helicopter, satellite, laser, diver, submarine and
flyers are all unshootable (the bullet passes straight through them).

## Player

- Land/water surface row: hull rides y119-122 (8x4 sprite, never mirrored).
- RIGHT +0.5 px/f up to the x=73 hard stop (the world scrolls past 0.25 px/f);
  LEFT −0.25 px/f down to x=8.
- Jump: fixed ballistic arc, ~135 frames airborne, apex +26px; holding or
  releasing UP does not change it. In water, DOWN submerges to y~139-142 with
  bobbing, UP climbs to y~94 and bobs back.
- Shot: one 1x4 bar in the hull color, from (hull_right+3, y115), flying
  (+2,−2) px/frame, despawning ~30 frames later at y~56. One in the air at a
  time, edge-triggered (no autofire).

## Enemies

**Helicopter** (land + Water A): enters right ~x150, flies left ~0.58 px/f,
slowing to ~0.27 px/f mid-screen; pass ~305 frames. Drops 0-2 (rarely 3)
bombs per pass, only when the horizontal gap to the player is ≤ ~75px; the
bomb falls (−1,+2) px/frame — straight down (0,+2) if released near the left
edge — and is **never** aimed rightward at the player. Bombs kill on contact
only. The yellow "wedge / diamond spray" behind it is a scripted, harmless
animation (once per pass, 88 frames from pass+103).

**Satellite** (land + Water A): enters left x=8, moves right exactly
+1,+1,+1,+0 (3px / 4 frames), pass ~204 frames, **44-48 frame gap** between
passes. Its laser (1x4, from belly+5, y90) falls straight down 1px/frame,
aim-once, never homing, indestructible.
- Land trigger: ~2 drops per pass at pseudo-random times (spacing 30-90
  frames), player-position independent.
- Water trigger (nailed by RAM injection): at discrete check moments it fires
  iff the drop column is **1..95px to the right of the player's hull** —
  never from behind. One laser in the air at a time.

**Laser splash / frogman** (Water A): the bolt keeps sinking under water to
y~134-137, then a green frogman surfaces at [laser_x, laser_x+19] straddling
the surface, riding the world scroll (0.25 px/f left). Kill window: boat x in
[laser_x−9, laser_x+19]; flying clears it, **diving does not**.

**Scuba diver** (Water A): a vertical 7x20 swimmer entering from the right at
depth y131-150, swimming left 0.25 px/f, animating every 15 frames, vanishing
mid-screen on an age clock (~333 frames clean). Dangerous to a diving boat.

**Water B roster** (all sprites cropped from real frames): the surface rocket
(idles riding the scroll, then ignites and launches skyward; +200 for ramming
it), the submarine cruising at y~137-147 (threat to a diving boat only), and
the pink helicopter (y57) and small red flyer (y61) crossing the sky.

## Known deviations / open questions

- The satellite's exact check-moment generator (gaps 10-60 frames, state
  dependent) is unidentified; the JAX version uses a fixed 30-frame check.
- The land helicopter/satellite pass scheduling follows a 256/315-frame slot
  lattice with skipped slots; the JAX version uses the simpler
  spawn-when-row-empty rule.
- Whether the boat can fire in the water scenes is contradicted between
  measurement runs (two agents measured the anti-air shot, one measured no
  projectile at all); the JAX version keeps the shot.
- Water B's exit condition (if any) is unknown.
- Lives are deducted at the END of the 59-frame animation in the real game;
  the JAX version deducts immediately and then freezes (kept for test
  compatibility).

The raw measurement reports, per-frame CSV traces and evidence frames live in
the session scratchpad (`reports/*.md`) if deeper numbers are ever needed.
