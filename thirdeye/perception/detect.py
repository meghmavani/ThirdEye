"""Part detection and tracking, and the step from pixels onto the stud grid
(FR-PER-2, FR-PER-3, FR-PER-4).

Two detectors behind one method, `detect(frame) -> list[Detection]`:

*`YoloDetector`* -- the fine-tuned model of plan 4.6, whose class names are the
part names (`red_2x4`). With `track=True` it runs ByteTrack through Ultralytics
and each detection carries a stable id (FR-PER-4); that is the whole of the
plan's `track.py`, so it lives here rather than in a module of its own.

*`ZeroShotDetector`* -- YOLO-World, the documented fallback and ablation
baseline (FR-PER-3). A text prompt can find "a lego brick" but cannot reliably
tell a 2x3 from a 2x4 or name a colour under desk lighting, so it is only asked
for the box. Size then comes from geometry -- the extent in studs of the
brick's own pixels through the homography (see `footprint_studs`) -- and colour
from hue. Both are measurable things, which is
why this is a fair baseline rather than a strawman.

`to_readings` is where either becomes what the tracker consumes: a `Reading`
on the grid, with its residual.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import cv2
import numpy as np

from thirdeye.compiler.schema import PartRef, Rotation
from thirdeye.state.lego.grid import Reading, StudGrid, rotation_from_extent


@dataclass(frozen=True)
class Detection:
    """One part in one frame, in image pixels."""

    box: tuple[float, float, float, float]      # x0, y0, x1, y1
    confidence: float
    #: None when the detector only found "a part" and geometry must name it.
    part: PartRef | None = None
    track_id: int | None = None


class Detector(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> list[Detection]: ...


def part_from_class(name: str) -> PartRef | None:
    """`red_2x4` -> PartRef. Anything else is not a part class."""
    color, _, size = name.partition("_")
    try:
        return PartRef(color=color, size=size)
    except ValueError:
        return None


# --- colour, for the zero-shot path ------------------------------------------------------

#: OpenCV hue (0-179) bands for the brick colours a classic box has. Grey and
#: black and white are told apart by saturation and value instead of hue.
HUES: Mapping[str, tuple[int, int]] = {
    "red": (0, 8),
    "orange": (9, 20),
    "yellow": (21, 34),
    "green": (35, 85),
    "blue": (86, 130),
    "purple": (131, 160),
    "pink": (161, 172),
}


def dominant_color(frame_bgr: np.ndarray, box: Sequence[float], inset: float = 0.25) -> str:
    """The brick colour in the middle of a box.

    The inset keeps the edges -- shadows, the plate, a neighbour -- out of the
    sample. Median rather than mean, so studs' highlights do not drag it.
    """
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * inset, (y1 - y0) * inset
    patch = frame_bgr[int(y0 + dy):int(y1 - dy) + 1, int(x0 + dx):int(x1 - dx) + 1]
    if patch.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h, s, v = (float(np.median(hsv[:, i])) for i in range(3))
    if v < 50:
        return "black"
    if s < 50:
        return "white" if v > 170 else "grey"
    if h >= 173:
        return "red"   # red wraps around the hue circle
    for name, (lo, hi) in HUES.items():
        if lo <= h <= hi:
            return name
    return "unknown"


# --- onto the grid ------------------------------------------------------------------------

def box_extent_studs(
    box: Sequence[float], homography: Sequence[Sequence[float]]
) -> tuple[float, float, float, float]:
    """(centre x, centre y, extent x, extent y) of a pixel box, in studs."""
    x0, y0, x1, y1 = box
    corners = np.array([[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]], dtype=np.float64)
    studs = cv2.perspectiveTransform(corners, np.asarray(homography, dtype=np.float64))
    studs = studs.reshape(-1, 2)
    lo, hi = studs.min(axis=0), studs.max(axis=0)
    centre = (lo + hi) / 2
    return float(centre[0]), float(centre[1]), float(hi[0] - lo[0]), float(hi[1] - lo[1])


def size_from_extent(ex: float, ey: float) -> tuple[str, Rotation]:
    """Stud extents -> (size, rotation), rounded to whole studs.

    A size is written short side first, so `2x4` spans 2 in x at rot 0 and 4
    in x at rot 90 -- the same convention the spec and `Pose.footprint` use.
    """
    w, h = max(1, round(ex)), max(1, round(ey))
    if w <= h:
        return f"{w}x{h}", 0
    return f"{h}x{w}", 90


def footprint_studs(
    frame_bgr: np.ndarray,
    box: Sequence[float],
    homography: Sequence[Sequence[float]],
    min_pixels: int = 30,
) -> tuple[float, float, float, float]:
    """(centre x, centre y, extent x, extent y) of the brick's own pixels, in studs.

    A detector's box is axis-aligned in the *image*. Under any camera rotation
    the brick sits diagonally inside it, and the box's corners map to stud
    points well outside the brick: a 2x4 measures as a 2x5 or a 3x4. So the
    brick is found inside the box by its colour -- pixels close to the colour at
    the box's centre -- and those pixels are mapped instead. Percentiles, not
    min and max, so a stray matching pixel on a neighbour does not stretch it.
    Falls back to the box when too few pixels match.
    """
    x0, y0, x1, y1 = (round(v) for v in box)
    h_img, w_img = frame_bgr.shape[:2]
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w_img), min(y1, h_img)
    patch = frame_bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return box_extent_studs(box, homography)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(np.int32)
    ph, pw = hsv.shape[:2]
    core = hsv[ph // 3: max(ph // 3 + 1, 2 * ph // 3), pw // 3: max(pw // 3 + 1, 2 * pw // 3)]
    med = np.median(core.reshape(-1, 3), axis=0)
    dv = np.abs(hsv[..., 2] - med[2])
    ds = np.abs(hsv[..., 1] - med[1])
    if med[1] < 50:          # achromatic: hue is noise, match on brightness only
        mask = (dv < 40) & (ds < 50)
    else:
        dh = np.abs(hsv[..., 0] - med[0])
        dh = np.minimum(dh, 180 - dh)
        mask = (dh < 10) & (ds < 80) & (dv < 80)
    ys, xs = np.nonzero(mask)
    if len(xs) < min_pixels:
        return box_extent_studs(box, homography)
    pts = np.stack([xs + x0 + 0.5, ys + y0 + 0.5], axis=1).reshape(-1, 1, 2).astype(np.float64)
    studs = cv2.perspectiveTransform(pts, np.asarray(homography, dtype=np.float64)).reshape(-1, 2)
    lo = np.percentile(studs, 1, axis=0)
    hi = np.percentile(studs, 99, axis=0)
    centre = (lo + hi) / 2
    return float(centre[0]), float(centre[1]), float(hi[0] - lo[0]), float(hi[1] - lo[1])


def to_readings(
    detections: Sequence[Detection],
    frame_bgr: np.ndarray,
    homography: Sequence[Sequence[float]],
    grid: StudGrid,
    layer: int = 0,
) -> list[Reading]:
    """Detections -> grid readings. Off-plate detections are simply absent.

    `layer` is 0 until the side camera supplies heights (FR-STA-5); Model A is
    designed flat so the must-have demo never needs more (plan 2.4).
    """
    out = []
    for det in detections:
        cx, cy, ex, ey = footprint_studs(frame_bgr, det.box, homography)
        if det.part is not None:
            part = det.part
            rot = rotation_from_extent(ex, ey, part)
        else:
            size, rot = size_from_extent(ex, ey)
            color = dominant_color(frame_bgr, det.box)
            if color == "unknown":
                continue
            part = PartRef(color=color, size=size)
        reading = grid.read(part, (cx, cy), rot=rot, layer=layer, confidence=det.confidence)
        if reading is not None:
            out.append(reading)
    return out


# --- the models ---------------------------------------------------------------------------

class YoloDetector:
    """A fine-tuned YOLO whose classes are part names (plan 4.6)."""

    def __init__(self, weights: str, conf: float = 0.25, track: bool = True,
                 imgsz: int = 640, device: str | None = None):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf, self.track, self.imgsz, self.device = conf, track, imgsz, device
        self.parts = {i: part_from_class(n) for i, n in self.model.names.items()}

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        kwargs: dict[str, Any] = {"conf": self.conf, "imgsz": self.imgsz, "verbose": False}
        if self.device:
            kwargs["device"] = self.device
        if self.track:
            results: Any = self.model.track(frame_bgr, persist=True,
                                            tracker="bytetrack.yaml", **kwargs)
        else:
            results = self.model.predict(frame_bgr, **kwargs)
        result = results[0]
        return _detections(result, self.parts)


class ZeroShotDetector:
    """YOLO-World asked only for boxes around bricks (FR-PER-3)."""

    PROMPTS = ("lego brick", "toy building brick")

    def __init__(self, weights: str = "yolov8s-worldv2.pt", conf: float = 0.1,
                 imgsz: int = 640, device: str | None = None):
        from ultralytics import YOLOWorld

        self.model = YOLOWorld(weights)
        self.model.set_classes(list(self.PROMPTS))
        self.conf, self.imgsz, self.device = conf, imgsz, device

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        kwargs: dict[str, Any] = {"conf": self.conf, "imgsz": self.imgsz, "verbose": False}
        if self.device:
            kwargs["device"] = self.device
        results: Any = self.model.predict(frame_bgr, **kwargs)
        return _detections(results[0], {})


def _detections(result, parts: Mapping[int, PartRef | None]) -> list[Detection]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(xyxy)
    return [
        Detection(
            box=tuple(float(v) for v in b),  # type: ignore[arg-type]
            confidence=float(c),
            part=parts.get(int(k)),
            track_id=None if i is None else int(i),
        )
        for b, c, k, i in zip(xyxy, conf, cls, ids, strict=True)
    ]
