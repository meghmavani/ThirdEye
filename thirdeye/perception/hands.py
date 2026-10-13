"""Hand keypoints and what they mean for the workspace (FR-PER-1, FR-PER-6).

MediaPipe finds 21 keypoints per hand; everything the rest of the system needs
is derived from them here, as plain geometry:

*Over the workspace.* Any keypoint inside the baseplate, mapped through the
calibration homography. This is the switch that freezes the LEGO tracker
(FR-PER-6), so it errs toward "yes": a fingertip over the edge of the plate
hides bricks just as surely as a palm over the middle.

*Grasping.* Thumb tip near index tip, relative to the size of the hand, so
it does not depend on how far the hand is from the camera.

*Hands in contact* -- the Jenga one-hand rule (FR-CHK-10). A hand counts if a
fingertip is inside the tower's image region.

The MediaPipe wrapper is deliberately thin: the model file is fetched, not
vendored (`scripts/fetch_models.sh`), and nothing below `HandDetector` needs it,
so the geometry is tested without a camera or a model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from thirdeye.state.lego.grid import apply_homography

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "hand_landmarker.task"

#: MediaPipe hand landmark indices.
WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, PINKY_MCP = 0, 4, 5, 8, 9, 17
FINGERTIPS = (4, 8, 12, 16, 20)


@dataclass(frozen=True)
class Hand:
    """One detected hand, in image pixels."""

    keypoints: np.ndarray       # (21, 2)
    score: float = 1.0
    handedness: str = ""

    @property
    def size_px(self) -> float:
        """Wrist to middle-finger knuckle: a scale that ignores finger pose."""
        return float(np.linalg.norm(self.keypoints[MIDDLE_MCP] - self.keypoints[WRIST]))


def over_workspace(
    hand: Hand, homography: Sequence[Sequence[float]], studs_x: int, studs_y: int,
    margin_studs: float = 0.5,
) -> bool:
    """Is any part of this hand over the baseplate?

    `margin_studs` widens the plate slightly: a hand hovering just off the edge
    still casts the occlusion we are guarding against.
    """
    for px, py in hand.keypoints:
        try:
            x, y = apply_homography(homography, float(px), float(py))
        except ValueError:
            continue
        if -margin_studs <= x <= studs_x + margin_studs and -margin_studs <= y <= studs_y + margin_studs:
            return True
    return False


def is_grasping(hand: Hand, ratio: float = 0.35) -> bool:
    """Thumb and index tips pinched together, relative to hand size."""
    size = hand.size_px
    if size <= 0:
        return False
    pinch = float(np.linalg.norm(hand.keypoints[THUMB_TIP] - hand.keypoints[INDEX_TIP]))
    return pinch / size < ratio


def hands_in_contact(hands: Sequence[Hand], region: np.ndarray) -> int:
    """How many hands have a fingertip inside `region`, a pixel polygon.

    For Jenga the region is the tower's outline in one side view; the count
    that reaches the checker is the most seen across both views and across the
    frames of a pull (see `TowerLattice.observe`).
    """
    poly = np.asarray(region, dtype=np.float32).reshape(-1, 1, 2)
    count = 0
    for hand in hands:
        if any(
            cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0
            for x, y in hand.keypoints[list(FINGERTIPS)]
        ):
            count += 1
    return count


class HandDetector:
    """MediaPipe Hand Landmarker in video mode (FR-PER-1). 5-10 ms per frame
    is the budget (NFR-PERF-4)."""

    def __init__(self, model: str | Path = DEFAULT_MODEL, max_hands: int = 2,
                 min_confidence: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        if not Path(model).exists():
            raise FileNotFoundError(f"{model} not found; run scripts/fetch_models.sh")
        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ms = -1

    def detect(self, frame_bgr: np.ndarray, t: float) -> list[Hand]:
        """Hands in one BGR frame taken at `t` seconds."""
        ms = max(int(t * 1000), self._last_ms + 1)   # video mode needs increasing stamps
        self._last_ms = ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect_for_video(
            self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb), ms
        )
        h, w = frame_bgr.shape[:2]
        hands = []
        for i, landmarks in enumerate(result.hand_landmarks):
            pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float64)
            cat = result.handedness[i][0] if result.handedness else None
            hands.append(Hand(pts, score=cat.score if cat else 1.0,
                              handedness=cat.category_name if cat else ""))
        return hands

    def close(self) -> None:
        self._landmarker.close()
