"""Print sheet for the four calibration markers (FR-CAL-1).

    python scripts/make_markers.py markers.png

Draws markers 0-3 at their true printed size on an A4 page at 300 DPI -- print
at 100% / "actual size", never "fit to page". Each is labelled with the plate
corner it belongs at: stick it just outside that corner, half a stud of gap,
as `MarkerLayout.around_plate` assumes. Rotation does not matter; position does.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from thirdeye.perception.calibrate import MarkerLayout
from thirdeye.state.lego.grid import STUD_PITCH_MM

DPI = 300
A4_MM = (210, 297)
CORNERS = {0: "bottom-left (origin)", 1: "bottom-right", 2: "top-right", 3: "top-left"}


def mm(v: float) -> int:
    return round(v / 25.4 * DPI)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--studs", type=int, default=16, help="baseplate size in studs")
    args = ap.parse_args(argv)

    layout = MarkerLayout.around_plate(args.studs, args.studs)
    side_mm = layout.side_studs * STUD_PITCH_MM
    page = np.full((mm(A4_MM[1]), mm(A4_MM[0])), 255, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(layout.dictionary)
    side = mm(side_mm)
    for i, mid in enumerate(sorted(layout.centres)):
        x = mm(30 + (i % 2) * 90)
        y = mm(40 + (i // 2) * 90)
        page[y:y + side, x:x + side] = cv2.aruco.generateImageMarker(dictionary, mid, side)
        cv2.putText(page, f"id {mid}: {CORNERS.get(mid, '')}", (x, y + side + mm(8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 3)
    cv2.putText(page, f"Print at 100%. Each marker must measure {side_mm:.0f} mm.",
                (mm(20), mm(20)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3)
    cv2.imwrite(str(args.out), page)
    print(f"wrote {args.out}: {side_mm:.0f} mm markers for a {args.studs}x{args.studs} plate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
