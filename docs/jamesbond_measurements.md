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

**Laser splash / frogman** (Water A): with no scuba diver on screen, the bolt
keeps sinking under water to y~134-137, then a green frogman surfaces at
[laser_x, laser_x+19] straddling the surface and riding the world scroll
(0.25 px/f left). Kill window: boat x in [laser_x−9, laser_x+19]; flying
clears it, **diving does not**. If a scuba diver is present, the spent bolt
instead disappears normally and creates no radioactive splash.

**Scuba diver** (Water A): a vertical 7x20 swimmer entering from the right at
depth y131-150, swimming left 0.25 px/f, animating every 15 frames, vanishing
mid-screen on an age clock (~333 frames clean). Dangerous to a diving boat.
The depth charge removes the diver for +200 (team decision, overriding the
earlier ALE note). When the visible gap between the player's boat and scuba is
at most 40 horizontal pixels, the diver changes immediately into the same
radioactive narrow/wide figure created by a satellite drop, drawn straddling
the waterline like that splash (row 123) and hittable there; this proximity
transition is independent of the satellite bolt. Team rules: only one thing
is radioactive at a time, and once the first diver has appeared in the scene
the satellite bolt never splashes again.

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

## Addendum: second measurement round (frame-exact probes)

- **"Satellite fires two shots" — refuted.** 3,162 frames, 9 passes, forced
  RAM probes: one downward laser max, never anything above its body. The
  illusion: the helicopter's dashed exhaust trail above the satellite while
  a laser falls below, the orange pod permanently hanging under its body,
  and the helicopter bomb wearing the satellite's colors (shared sprite
  slot) while crossing its rows.
- **Splash frogman animation**: narrow pose (explosion_1) for exactly 1
  frame, then strict 7-frame phases alternating starting wide
  (explosion_2, anchored 4px left, carrying the yellow under-glow rows).
  120 frames exactly, world-fixed, instant removal. Never blinks.
- **The "radioactive bolt"**: a bolt landing while a frogman lives spawns
  nothing new — it sinks below the waterline recolored green, and the
  living frogman's despawn clock restarts (life extended to landing+120).
  A scuba diver suppresses the bolt-to-frogman conversion entirely. The diver
  itself changes to its radioactive state when the player's boat is nearby,
  regardless of whether a bolt is airborne or landing.
- **Water-B rocket cycle (256 frames)**: floats submerged (tip ~y140),
  climbs exactly 1px/frame, single-frame splash blip at the waterline,
  explodes at tip y61 into two red debris bars (the "red flyer" of the
  first survey — not an enemy) plus a 1-2 frame full-sky gray flash.
- **Submarine torpedo**: previously undocumented — a short (-2,-1)/frame
  underwater dart from the submarine, dies before the surface, harmless
  in all 40 probe branches.
- Helicopter bombs landing in water spawn nothing at all.

## Addendum: third water scene ("water C", daylight) — from video only

> Reference only: the project stops at water B, so this scene is not
> implemented. The notes stay for whoever picks it up later.

Source: a scrubbed YouTube longplay (`Screen Recording 2026-08-13 at
23.17.30.mov`, scene at 1:47–3:00), NOT ALE. Rows are calibrated on the
waterline (121) and are good to ~3 rows; speeds come from 5–60 fps
samples. The water-B exit was never seen (the video cuts from a water-B
death straight into this scene with +5000 on the counter), so the JAX
version hands over on a clock (`STAGE_WB_LENGTH`).

| element | video |
|---|---|
| sky | solid blue (70,80,207) down to the waterline, no stars |
| clouds | white 17x9, two per 160px strip ~62px apart (low ~row 36, high ~row 23), scrolling **0.5 px/f** — twice the seabed |
| water / seabed | (15,47,144); 160px hill strip, green over brown, 0.25 px/f |
| rocket pads | small gray pyramid (5x8, 5-row flame while climbing); **up to two** in the water at random columns, riding the scroll; climb 1 px/f (still drifting with the world), burst near row 61 |
| debris | falls ~1 row/f back to the waterline (the anti-air shot pops it there for **+100**, video 13500→13600), then floats ~40 f as a red/pink sparkle of 2-5 dots on a 6x3 footprint, two patterns swapping every ~6 f, and vanishes; treated as lethal |
| shots | anti-air (+2,−2) kills a climbing rocket, the sinking depth charge kills a submerged/surfacing one: **+100** each; the depth charge sinks the submarine: **+200**. Only these scored in 70 s |
| submarine | enters from the **left**, cruises right ~0.7 px/f (implemented 2px/3f) at ~row 146; its shot (see the water-B addendum) is shared with water B |
| helicopter | red land heli, ~0.6 px/f constant, back-to-back passes; its bomb keeps sinking to ~row 140 |
| steamship | second phase only (rockets stop ~2800 f in): 16 wide on the waterline, red hull / yellow deck / gray funnel + smoke, 0.6 px/f right→left, back-to-back; player dived under it (lethality assumed) |
| absent | satellite, diamond, scuba, frogman, oil rig, pink heli, stars |
| ending | after ~3960 f the hills stop, a 31x26 orange base scrolls in (0.25 px/f); when it reaches ~x66 a green 8x18 objective appears ~20px right of its left edge near row 144, drifting left only ~0.05 px/f; diving into it freezes the scene for the death-style colour cycle, then **+5000** and the title screen (3 lives still shown) |

Easy misreading, checked at 60 fps and NOT in the footage: a
submarine-launched shot that rises to the surface and splits into two
bombs. The yellow bar rising above the submarine is the helicopter's
bomb passing under water; the dark bar is the player's own depth charge.

## Addendum: how water B ends (second recording, 60 fps read)

Scene names by stage index: [0] land, [1] water A / oil rig, [2] water B
(dark, big rockets, pink balls), [3] water C (daylight). The old note
"Water B never ended in a 22k-frame probe" was because nobody shot the
balls.

- The **pink ball** (9x11, two poses alternating) enters at the **left**
  edge at row 57 and crosses right at **1.75 px/f** (7 px / 4 f), on a
  ~250-frame cycle. The anti-air shot pops it: **+500**.
- The footage shows three pops (6000→6500, 6700→7200, 7600→8100) before
  the exit; on the last one the ball stays on screen in its striped pose,
  the world freezes for **59 frames** while the boat colour-cycles and
  the sky strobes dark/light for the first ~21 frames, then **+5000**
  (8100→13100) and the daylight scene starts. `WB_BALL_HITS_TO_EXIT`
  holds the count (3 here; the team's reading was 2).
- The same freeze + 5000 pattern closes every scene (rig landing,
  ball count, daylight objective).
- The water-B rocket is shootable: in the third recording the anti-air shot
  touches the climbing rocket at ~row 97 and it vanishes with **+200**
  (6500→6700). Rams still pay the same 200.

## Addendum: the submarine's shot (third recording, water B)

Applies to water B and the daylight scene alike. The submarine enters at
the **left** and cruises right (~0.6 px/f) in water B too, so the old
right-to-left dart in the code was wrong. As it passes mid-screen it fires
**two yellow 2x2 dots stacked with a one-row gap** (2x5) from its bow: the
pair runs back and up diagonally until it sits just under the waterline
(~row 124), then straight left along that row at ~1.3 px/f (implemented
4 px / 3 f) until it leaves the screen. It passes under a surfaced hull
and crosses a diving boat's path. The earlier "torpedo that splits into
two red bombs" was a misreading and has been replaced by this.
