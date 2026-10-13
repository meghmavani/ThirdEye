"""Running a procedure from a YAML config (FR-SYS-2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from thirdeye import run

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_switching_procedure_is_a_different_config_not_different_code(capsys) -> None:
    assert run.main([str(CONFIGS / "replay_jenga.yaml")]) == 0
    jenga = capsys.readouterr().out
    assert "jenga_game" in jenga and "game_over" in jenga

    assert run.main([str(CONFIGS / "replay_jigsaw.yaml")]) == 0
    jigsaw = capsys.readouterr().out
    assert "jigsaw_forest_12" in jigsaw and "out_of_order" in jigsaw


def test_the_committed_live_configs_load_and_resolve_their_paths() -> None:
    for name, mode in (("jigsaw", "jigsaw"), ("jenga", "jenga")):
        cfg = run.load(CONFIGS / f"{name}.yaml")
        got_mode, argv = run.argv_for(cfg, CONFIGS)
        assert got_mode == mode and "--spec" in argv
        spec = Path(argv[argv.index("--spec") + 1])
        assert spec.is_absolute() and spec.exists()


def test_the_jenga_config_passes_one_camera_dir_per_camera() -> None:
    _, argv = run.argv_for(run.load(CONFIGS / "jenga.yaml"), CONFIGS)
    assert argv.count("--camera-dir") == 2


def test_an_unknown_option_is_an_error_not_a_silent_default(tmp_path: Path, capsys) -> None:
    bad = tmp_path / "c.yaml"
    bad.write_text("mode: jigsaw\nspec: s.json\nbox: b.png\ncamra: 1\n")
    assert run.main([str(bad)]) == 2
    assert "camra" in capsys.readouterr().err


def test_a_missing_or_unknown_mode_is_refused(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("spec: x\n")
    (tmp_path / "b.yaml").write_text("mode: knitting\n")
    for name in ("a", "b"):
        with pytest.raises(run.ConfigError):
            run.load(tmp_path / f"{name}.yaml")


def test_set_overrides_an_option_for_one_run() -> None:
    cfg = run.load(CONFIGS / "jigsaw.yaml", run._parse_set(["camera=2", "no_hands=true"]))
    _, argv = run.argv_for(cfg, CONFIGS)
    assert argv[argv.index("--camera") + 1] == "2" and "--no-hands" in argv


def test_jenga_without_cameras_is_a_config_error() -> None:
    with pytest.raises(run.ConfigError, match="camera_dirs"):
        run.argv_for({"mode": "jenga", "spec": "s.json"}, Path("."))
