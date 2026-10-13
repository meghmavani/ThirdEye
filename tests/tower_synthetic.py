"""A synthetic RGB-D camera looking at a Jenga tower, with known ground truth.

The tower is a set of axis-aligned 75 x 25 x 15 mm boxes; the camera is the same
`PinholeCamera` the slot reader uses, so a rendered depth map is exactly what that
camera would measure of the tower. Rays are cast against the boxes (slab method),
a table plane and a far backdrop; depth is the forward distance, as a LiDAR depth map
reports it. Sensor noise and dropouts are optional, and a "hand" is a box floating
in front of the tower.

Colour is a crude shading: faces lit, anything strictly inside the tower's footprint
(tunnel walls and floors) dark -- enough for the colour cue to have something to read.
"""

from __future__ import annotations

import numpy as np

from thirdeye.perception.camera_model import PinholeCamera
from thirdeye.perception.depth import RGBDFrame

W, H = 256, 192
K = np.array([[270.0, 0, W / 2], [0, 270.0, H / 2], [0, 0, 1]])
SPAN = 75.0
CENTRE = (SPAN / 2, SPAN / 2)
BACKDROP_MM = 1500.0


def corner_camera(sign_x: int, sign_y: int, distance: float = 520.0, z: float = 160.0):
    """A camera at the diagonal corner (sign_x, sign_y), looking at the tower."""
    d = distance / np.sqrt(2)
    pos = (CENTRE[0] + sign_x * d, CENTRE[1] + sign_y * d, z)
    return PinholeCamera.look_at(pos, (CENTRE[0], CENTRE[1], z), K, (W, H))


def tower_boxes(
    occupancy: list[list[int]], pushed: dict[tuple[int, int], float] | None = None
) -> np.ndarray:
    """(n, 6) boxes [x0, y0, z0, x1, y1, z1] for the filled slots. `pushed` slides a
    block out along its length toward the low-coordinate end, by that many mm."""
    pushed = pushed or {}
    boxes = []
    for layer, row in enumerate(occupancy):
        for slot, filled in enumerate(row):
            if not filled:
                continue
            z0, z1 = layer * 15.0, layer * 15.0 + 15.0
            a0, a1 = slot * 25.0, slot * 25.0 + 25.0
            shift = pushed.get((layer, slot), 0.0)
            if layer % 2 == 0:
                boxes.append([0 - shift, a0, z0, SPAN - shift, a1, z1])
            else:
                boxes.append([a0, 0 - shift, z0, a1, SPAN - shift, z1])
    return np.array(boxes, dtype=np.float64).reshape(-1, 6)


def _cast(camera: PinholeCamera, boxes: np.ndarray):
    """Nearest hit per pixel: (depth mm, shade 0..1)."""
    rays = camera.rays()                                  # (h, w, 3), z-depth parameterised
    origin = camera.c
    best = np.full((camera.height, camera.width), np.inf)
    shade = np.zeros_like(best)

    # table plane
    with np.errstate(divide="ignore", invalid="ignore"):
        t_table = np.where(rays[..., 2] < 0, -origin[2] / rays[..., 2], np.inf)
    best = np.where(t_table > 0, t_table, np.inf)
    shade = np.where(np.isfinite(best), 0.6, 0.0)

    corners_of = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    for x0, y0, z0, x1, y1, z1 in boxes:
        lo = np.array([x0, y0, z0])
        hi = np.array([x1, y1, z1])
        # Only the pixels the box can cover: its projected bounding rectangle.
        px, z = camera.project(lo + corners_of * (hi - lo))
        if (z <= 1).any():
            r0, r1, c0, c1 = 0, camera.height, 0, camera.width
        else:
            c0, c1 = max(int(px[:, 0].min()) - 2, 0), min(int(px[:, 0].max()) + 3, camera.width)
            r0, r1 = max(int(px[:, 1].min()) - 2, 0), min(int(px[:, 1].max()) + 3, camera.height)
        if c0 >= c1 or r0 >= r1:
            continue
        sub = rays[r0:r1, c0:c1]
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (lo - origin) / sub
            t2 = (hi - origin) / sub
            tmin = np.minimum(t1, t2)
            tmax = np.maximum(t1, t2)
            near = np.nanmax(tmin, axis=-1)
            far = np.nanmin(tmax, axis=-1)
            cur = best[r0:r1, c0:c1]
            hit = (far >= near) & (far > 0) & (near > 0) & (near < cur)
            if not hit.any():
                continue
            point = origin + sub * near[..., None]
        inside = ((point[..., 0] > 2) & (point[..., 0] < SPAN - 2) &
                  (point[..., 1] > 2) & (point[..., 1] < SPAN - 2))
        best[r0:r1, c0:c1] = np.where(hit, near, cur)
        shade[r0:r1, c0:c1] = np.where(hit, np.where(inside, 0.35, 0.85), shade[r0:r1, c0:c1])
    return best, shade


def render(
    camera: PinholeCamera,
    occupancy: list[list[int]],
    t: float = 0.0,
    pushed: dict[tuple[int, int], float] | None = None,
    noise_mm: float = 0.0,
    dropout: float = 0.0,
    hand: tuple[float, float, float, float, float, float] | None = None,
    seed: int = 0,
) -> RGBDFrame:
    boxes = tower_boxes(occupancy, pushed)
    if hand is not None:
        boxes = np.vstack([boxes, np.array([hand], dtype=np.float64)])
    depth, shade = _cast(camera, boxes)
    depth = np.where(np.isfinite(depth), np.minimum(depth, BACKDROP_MM), BACKDROP_MM)
    rng = np.random.default_rng(seed)
    if noise_mm:
        depth = depth + rng.normal(0, noise_mm, depth.shape)
    if dropout:
        depth = np.where(rng.random(depth.shape) < dropout, 0.0, depth)
    gray = np.clip(np.where(shade == 0, 0.1, shade) * 255, 0, 255).astype(np.uint8)
    color = np.repeat(gray[..., None], 3, axis=2)
    return RGBDFrame(t=t, color=color, depth=(depth / 1000.0).astype(np.float32), k=camera.k)


def full(layers: int = 18) -> list[list[int]]:
    return [[1, 1, 1] for _ in range(layers)]


def without(occ: list[list[int]], *cells: tuple[int, int]) -> list[list[int]]:
    out = [list(r) for r in occ]
    for layer, slot in cells:
        out[layer][slot] = 0
    return out


def with_block(occ: list[list[int]], *cells: tuple[int, int]) -> list[list[int]]:
    out = [list(r) for r in occ]
    for layer, slot in cells:
        while len(out) <= layer:
            out.append([0, 0, 0])
        out[layer][slot] = 1
    return out
