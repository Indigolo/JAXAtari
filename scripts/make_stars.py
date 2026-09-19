#!/usr/bin/env python3
"""Generate the water-scene star field sprites (stars_1.npy / stars_2.npy).

Scattered triangle-of-3 clusters across the sky, mixed orientations, static
(both frames identical). Run from the repo root:  python scripts/make_stars.py
Tuning: edit `clusters` below. Each entry is (cx, cy, orient):
  orient 0 = point down (pair on top)      2 = point right (pair on left)
  orient 1 = point up   (pair on bottom)   3 = point left  (pair on right)
"""
import numpy as np

H, W = 40, 152
WHITE = np.array([236, 236, 236, 255], dtype=np.uint8)
OUT_1 = 'src/jaxatari/jb_sprites/stars_1.npy'
OUT_2 = 'src/jaxatari/jb_sprites/stars_2.npy'

def put(img, x, y):
    if 0 <= x < W and 0 <= y < H:
        img[y, x] = WHITE

def tri_dots(cx, cy, orient):
    if orient == 0:   # point down: pair on top, one below
        return [(cx-3, cy), (cx+3, cy), (cx, cy+5)]
    if orient == 1:   # point up: one on top, pair below
        return [(cx, cy), (cx-3, cy+5), (cx+3, cy+5)]
    if orient == 2:   # point right: pair on left, one right
        return [(cx, cy-3), (cx, cy+3), (cx+5, cy)]
    return              [(cx, cy-3), (cx, cy+3), (cx-5, cy)]  # point left

# (cx, cy, orient) — hand-placed, scattered heights and mixed orientations.
clusters = [
    (16, 8,  0), (44, 16, 1), (68, 6,  3),
    (92, 14, 0), (116, 9, 2), (138, 18, 1),
    (10, 22, 2), (34, 4,  1), (78, 24, 0), (126, 28, 3),
]

def main():
    f = np.zeros((H, W, 4), dtype=np.uint8)
    for (cx, cy, orient) in clusters:
        for (x, y) in tri_dots(cx, cy, orient):
            put(f, x, y)
    # Static: both frames identical (no twinkle).
    np.save(OUT_1, f)
    np.save(OUT_2, f.copy())
    print('wrote %d static triangles -> %s, %s' % (len(clusters), OUT_1, OUT_2))

if __name__ == '__main__':
    main()
