"""Colour + depth input for the Jenga rig (FR-CAL-9, SW-14, NFR-PERF-12).

An `RGBDFrame` is what everything downstream sees, wherever it came from: a colour
image, a depth map in **metres** the same size as the colour image (forward
distance, 0 or NaN where the sensor returned nothing), the intrinsics for that
size, and a timestamp. Three sources produce them:

*`RecordedRGBD`* -- a directory of `.npz` frames. Development and the evaluation
run on these, so the phone is not needed to build or test anything (FR-CAL-9), and
a labelled recording is just a directory with timestamps.

*`Record3DSource`* -- the live iPhone LiDAR stream over USB, through the
`record3d` package. Wired up to that package's documented streaming interface but
**not exercised against a phone** in this repository: OQ-8 (the access route) is
still open, and the first session on real hardware should treat it as a spike.

*`SyntheticSource`* in the tests -- rays cast against a model of the tower.

Depth is resized to the colour image with nearest-neighbour, never interpolated:
averaging a block's front face with the far wall behind it invents a surface that
is not there, which is exactly the signal the slot reader is looking for.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class RGBDFrame:
    t: float
    color: np.ndarray        # (h, w, 3) BGR uint8
    depth: np.ndarray        # (h, w) float32 metres; <= 0 or NaN where invalid
    k: np.ndarray            # (3, 3) intrinsics for (h, w)

    @property
    def size(self) -> tuple[int, int]:
        return self.color.shape[1], self.color.shape[0]

    def depth_mm(self) -> np.ndarray:
        """Depth in millimetres, invalid samples as NaN."""
        d = self.depth.astype(np.float32) * 1000.0
        d[~np.isfinite(d) | (d <= 0)] = np.nan
        return d


def aligned(color: np.ndarray, depth: np.ndarray, k_color: np.ndarray, t: float) -> RGBDFrame:
    """Build a frame from a colour image and a depth map of a different size.

    `k_color` are the intrinsics of the colour image.
    """
    h, w = color.shape[:2]
    if depth.shape[:2] != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    return RGBDFrame(t=t, color=color, depth=depth.astype(np.float32), k=np.asarray(k_color))


class RGBDSource(Protocol):
    def __iter__(self) -> Iterator[RGBDFrame]: ...


def save_frame(path: str | Path, frame: RGBDFrame) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, t=frame.t, color=frame.color, depth=frame.depth, k=frame.k)
    return path


def load_frame(path: str | Path) -> RGBDFrame:
    with np.load(path) as z:
        return RGBDFrame(t=float(z["t"]), color=z["color"], depth=z["depth"], k=z["k"])


class RecordedRGBD:
    """Frames from a directory of `.npz` files, in name order (FR-CAL-9)."""

    def __init__(self, directory: str | Path):
        self.paths = sorted(Path(directory).glob("*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"no .npz frames in {directory}")

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self) -> Iterator[RGBDFrame]:
        for p in self.paths:
            yield load_frame(p)


class Record3DSource:
    """The iPhone LiDAR stream over USB, via the `record3d` package.

    Not exercised on hardware here -- see the module docstring. Frames are timed by
    the host clock at arrival, which is what `sync.py` aligns across cameras; the
    phone's own timestamps are not used because the second camera has none.
    """

    def __init__(self, device_index: int = 0, timeout_s: float = 10.0):
        try:
            from record3d import Record3DStream
        except ImportError as exc:  # pragma: no cover - needs the package and a phone
            raise RuntimeError(
                "the live iPhone stream needs the `record3d` package "
                "(pip install record3d); for development use RecordedRGBD"
            ) from exc
        import threading
        import time

        self._time = time
        self._event = threading.Event()
        self._stream = Record3DStream()
        self._stream.on_new_frame = self._event.set
        self._stream.on_stream_stopped = lambda: None
        devices = Record3DStream.get_connected_devices()
        if device_index >= len(devices):
            raise RuntimeError(f"no iPhone found over USB ({len(devices)} device(s) seen)")
        self._stream.connect(devices[device_index])
        self._timeout = timeout_s
        self._start = time.monotonic()

    def __iter__(self) -> Iterator[RGBDFrame]:  # pragma: no cover - needs a phone
        while self._event.wait(self._timeout):
            self._event.clear()
            depth = np.asarray(self._stream.get_depth_frame())
            rgb = np.asarray(self._stream.get_rgb_frame())
            intr = self._stream.get_intrinsic_mat()
            k = np.array([[intr.fx, 0, intr.tx], [0, intr.fy, intr.ty], [0, 0, 1]], float)
            yield aligned(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), depth, k,
                          self._time.monotonic() - self._start)
