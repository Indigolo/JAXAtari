"""Step an ALE ROM one frame at a time and log where every object is.

Companion to ALE_RAMStateDeltas.py: that one watches RAM, this one watches
pixels. Used to measure movement constants (velocities, spawn offsets, drop
periods) off the real ROM instead of guessing them.

Typical session, measuring how a projectile falls:

    # 1. find the frame where the thing you care about appears
    python scripts/ale_frame_probe.py -g Jamesbond -n 600 --actions 0:FIRE

    # 2. re-run to that frame, checkpoint it, then step in detail
    python scripts/ale_frame_probe.py -g Jamesbond -n 200 --save-state heli.pkl
    python scripts/ale_frame_probe.py -g Jamesbond --load-state heli.pkl -n 60 --sprite 1x4

Checkpoints make a measurement repeatable: the same checkpoint plus the same
actions replays the exact same frames every time, so a number you read once
can be read again.
"""

import argparse
import csv
import pickle
import sys
from collections import Counter

import numpy as np

try:
    import gymnasium as gym
    import ale_py
except ImportError:  # pragma: no cover - dependency hint only
    sys.exit("needs gymnasium and ale-py: pip install -e '.[dev]'")

from scipy import ndimage


def make_env(game, seed):
    """Deterministic single-frame-stepping env: no frameskip, no sticky actions."""
    gym.register_envs(ale_py)
    try:
        env = gym.make(
            f"ALE/{game}-v5",
            frameskip=1,               # one env.step() == one console frame
            repeat_action_probability=0.0,  # no sticky actions, replays are exact
            render_mode="rgb_array",
        )
    except Exception as exc:
        sys.exit(
            f"could not load ALE/{game}-v5: {exc}\n"
            "the ROM has to sit next to the other .bin files in ale_py/roms/"
        )
    env.reset(seed=seed)
    return env


def parse_actions(spec, meanings):
    """'0:FIRE,120-300:RIGHT' -> a frame -> action-index lookup."""
    plan = {}
    if not spec:
        return plan
    for part in spec.split(","):
        span, _, name = part.partition(":")
        name = name.strip().upper()
        if name not in meanings:
            sys.exit(f"action {name!r} not in this game: {meanings}")
        idx = meanings.index(name)
        start, _, end = span.partition("-")
        for frame in range(int(start), int(end or start) + 1):
            plan[frame] = idx
    return plan


def find_objects(frame, background, min_pixels):
    """Bounding box of every connected blob of one colour.

    ALE draws each object in its own colour, so grouping pixels by colour and
    then splitting into connected components recovers the sprites without
    needing to know what they look like.
    """
    flat = frame.reshape(-1, 3)
    objects = []
    for colour in np.unique(flat, axis=0):
        if tuple(int(c) for c in colour) in background:
            continue
        mask = np.all(frame == colour, axis=-1)
        labels, count = ndimage.label(mask)
        for ys, xs in ndimage.find_objects(labels):
            height = ys.stop - ys.start
            width = xs.stop - xs.start
            if mask[ys, xs].sum() < min_pixels:
                continue
            objects.append(
                {
                    "x": xs.start,
                    "y": ys.start,
                    "w": width,
                    "h": height,
                    "colour": "#%02x%02x%02x" % tuple(int(c) for c in colour),
                }
            )
    return objects


def guess_background(frame, keep=3):
    """The few most common colours are the sky, the ground and the HUD."""
    flat = frame.reshape(-1, 3)
    counts = Counter(map(tuple, flat.tolist()))
    return {colour for colour, _ in counts.most_common(keep)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-g", "--game", default="Jamesbond")
    ap.add_argument("-n", "--frames", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--actions", default="", help="e.g. '0:FIRE,120-300:RIGHT'")
    ap.add_argument("--sprite", default="", help="only report this size, e.g. 1x4")
    ap.add_argument("--min-pixels", type=int, default=2)
    ap.add_argument("--csv", help="write every object of every frame here")
    ap.add_argument("--ram", action="store_true", help="print changed RAM bytes")
    ap.add_argument("--save-state", help="pickle the console state after the run")
    ap.add_argument("--load-state", help="start from a pickled console state")
    args = ap.parse_args()

    env = make_env(args.game, args.seed)
    ale = env.unwrapped.ale
    meanings = env.unwrapped.get_action_meanings()
    plan = parse_actions(args.actions, meanings)

    if args.load_state:
        with open(args.load_state, "rb") as fp:
            ale.restoreState(pickle.load(fp))

    want = None
    if args.sprite:
        w, _, h = args.sprite.partition("x")
        want = (int(w), int(h))

    background = guess_background(env.render())
    previous_ram = ale.getRAM().copy()
    rows = []

    for frame_no in range(args.frames):
        action = plan.get(frame_no, 0)
        _, _, terminated, truncated, _ = env.step(action)
        frame = env.render()

        objects = find_objects(frame, background, args.min_pixels)
        if want:
            objects = [o for o in objects if (o["w"], o["h"]) == want]

        for obj in objects:
            obj["frame"] = frame_no
            rows.append(obj)
            print(
                f"f{frame_no:5d}  x={obj['x']:3d} y={obj['y']:3d} "
                f"{obj['w']}x{obj['h']}  {obj['colour']}"
            )

        if args.ram:
            ram = ale.getRAM()
            changed = np.nonzero(ram != previous_ram)[0]
            if changed.size:
                deltas = " ".join(
                    f"{i}:{previous_ram[i]}->{ram[i]}" for i in changed
                )
                print(f"f{frame_no:5d}  RAM {deltas}")
            previous_ram = ram.copy()

        if terminated or truncated:
            print(f"episode ended at frame {frame_no}")
            break

    if args.save_state:
        with open(args.save_state, "wb") as fp:
            pickle.dump(ale.cloneState(), fp)
        print(f"state saved to {args.save_state}")

    if args.csv and rows:
        with open(args.csv, "w", newline="") as fp:
            writer = csv.DictWriter(
                fp, fieldnames=["frame", "x", "y", "w", "h", "colour"]
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"{len(rows)} rows written to {args.csv}")

    env.close()


if __name__ == "__main__":
    main()
