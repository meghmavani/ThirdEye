"""Score the scripted scenarios: precision, recall and false alerts per procedure.

    python -m eval.run_scenarios [scenario dir] [--json out.json]

Each script under `scripts/scenarios/` carries the alerts a correct checker raises
(`expect:`), which are the planted errors and their direct consequences. Replaying
every script through the real tracker and checker and scoring the alerts against
those labels is the dry run of the protocol that real recordings follow (plan 5.1,
5.2): when labelled recordings exist, their labels take the place of `expect:` and
this script is unchanged.

Scripts are grouped by the procedure their spec declares. The scenarios are scripted
*scenes*, so this scores the tracker and checker and not perception; the perception
numbers (EV-STA-*) come from `eval.perception_synthetic` and, later, real clips.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from eval.metrics import Planted, Run, detection_table, score_detection
from thirdeye.replay import Replay

SCENARIOS = Path(__file__).resolve().parents[1] / "scripts" / "scenarios"


def _planted(expect: list[str]) -> list[Planted]:
    out = []
    for item in expect:
        code, _, target = item.partition(" ")
        out.append(Planted(code=code, target=target.strip()))
    return out


def run_all(directory: Path = SCENARIOS) -> dict[str, list[Run]]:
    groups: dict[str, list[Run]] = defaultdict(list)
    for path in sorted(directory.glob("*.yaml")):
        script = yaml.safe_load(path.read_text(encoding="utf-8"))
        record = Replay.load(path).run()
        groups[record.procedure].append(
            Run(path.stem, _planted(script.get("expect", [])), record.alerts)
        )
    return dict(groups)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("directory", nargs="?", type=Path, default=SCENARIOS)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    summary = {}
    for procedure, runs in run_all(args.directory).items():
        d = score_detection(runs)
        print(f"## {procedure} ({len(runs)} scripts)\n")
        print(detection_table(d))
        print()
        summary[procedure] = {
            "runs": [r.name for r in runs],
            "precision": d.overall().precision,
            "recall": d.overall().recall,
            "false_alerts_per_correct_run": d.false_alerts_per_correct_run,
            "by_code": {c: vars(s) for c, s in d.by_code.items()},
        }
    if args.json:
        args.json.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
