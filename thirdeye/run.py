"""Run a procedure from a YAML config (FR-SYS-2).

    python -m thirdeye.run configs/jigsaw.yaml
    python -m thirdeye.run configs/jenga.yaml --set serve=8000
    python -m thirdeye.run configs/replay_jenga.yaml

A config names the `mode` -- which loop to run -- and that mode's options; switching
between Jenga and jigsaw is a different file, not different code. Paths in a config are
relative to the config file. Unknown keys are an error: a misspelt `camera_dirs` that
silently fell back to a default would run the wrong rig during a demo.

    mode: jigsaw     spec, box, [camera | video], [no_hands], [serve], [log], [seconds]
    mode: jenga      spec, camera_dirs (one per camera), [rig | save_rig], [offset],
                     [pose_frame], [no_hands], [serve], [log]
    mode: replay     script, [log]
    mode: lego       spec, (weights | zero_shot), [camera | video], ...  (parked)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

#: option -> (flag, takes a value). Per mode, in the order they are emitted.
OPTIONS: dict[str, dict[str, tuple[str, bool]]] = {
    "jigsaw": {
        "spec": ("--spec", True), "box": ("--box", True), "camera": ("--camera", True),
        "video": ("--video", True), "no_hands": ("--no-hands", False),
        "no_show": ("--no-show", False), "serve": ("--serve", True), "log": ("--log", True),
        "seconds": ("--seconds", True),
    },
    "jenga": {
        "spec": ("--spec", True), "rig": ("--rig", True), "save_rig": ("--save-rig", True),
        "offset": ("--offset", True), "pose_frame": ("--pose-frame", True),
        "no_hands": ("--no-hands", False), "serve": ("--serve", True), "log": ("--log", True),
    },
    "replay": {"script": ("", True), "log": ("--log", True), "quiet": ("--quiet", False)},
    "lego": {
        "spec": ("--spec", True), "weights": ("--weights", True), "zero_shot": ("--zero-shot", False),
        "camera": ("--camera", True), "video": ("--video", True), "no_hands": ("--no-hands", False),
        "no_show": ("--no-show", False), "seconds": ("--seconds", True), "log": ("--log", True),
    },
}
#: Options that are paths, resolved against the config file.
PATHS = {"spec", "box", "video", "rig", "save_rig", "log", "script", "weights"}


class ConfigError(ValueError):
    pass


def load(path: Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or "mode" not in cfg:
        raise ConfigError(f"{path}: a config is a mapping with a 'mode'")
    cfg = {**cfg, **(overrides or {})}
    mode = cfg["mode"]
    if mode not in OPTIONS:
        raise ConfigError(f"{path}: mode {mode!r} is not one of {sorted(OPTIONS)}")
    allowed = set(OPTIONS[mode]) | {"mode"} | ({"camera_dirs"} if mode == "jenga" else set())
    unknown = set(cfg) - allowed
    if unknown:
        raise ConfigError(f"{path}: unknown option(s) {sorted(unknown)} for mode {mode!r}")
    return cfg


def argv_for(cfg: dict[str, Any], base: Path) -> tuple[str, list[str]]:
    """The mode and the command-line its loop is run with."""
    mode = cfg["mode"]
    argv: list[str] = []

    def resolve(key: str, value: Any) -> str:
        return str((base / str(value)).resolve()) if key in PATHS else str(value)

    for key, (flag, takes) in OPTIONS[mode].items():
        if key not in cfg:
            continue
        value = cfg[key]
        if not takes:
            if value:
                argv.append(flag)
            continue
        if flag:
            argv += [flag, resolve(key, value)]
        else:
            argv.append(resolve(key, value))
    if mode == "jenga":
        dirs = cfg.get("camera_dirs") or []
        if not dirs:
            raise ConfigError("mode jenga needs camera_dirs, one per camera")
        for d in dirs:
            argv += ["--camera-dir", str((base / d).resolve())]
    return mode, argv


def _parse_set(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise ConfigError(f"--set wants key=value, got {item!r}")
        out[key] = yaml.safe_load(value)
    return out


def dispatch(mode: str, argv: list[str]) -> int:
    if mode == "jigsaw":
        from thirdeye.live_jigsaw import main
    elif mode == "jenga":
        from thirdeye.live_jenga import main
    elif mode == "replay":
        from thirdeye.replay import main
    else:
        from thirdeye.live import main
    return main(argv)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", type=Path)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config option for this run")
    args = ap.parse_args(argv)
    try:
        cfg = load(args.config, _parse_set(args.set))
        mode, cli = argv_for(cfg, args.config.resolve().parent)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    return dispatch(mode, cli)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
