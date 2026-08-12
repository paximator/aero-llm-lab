from pathlib import Path

from aerollm.generation.smoke import _config_digest


def test_config_digest_is_stable_and_ignores_non_json_files(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text('{"model":"ministral"}', encoding="utf-8")
    first = _config_digest(tmp_path)

    (tmp_path / "weights.bin").write_bytes(b"ignored")

    assert _config_digest(tmp_path) == first


def test_config_digest_includes_json_names_and_content(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    first = _config_digest(tmp_path)
    (tmp_path / "generation_config.json").write_text("{}", encoding="utf-8")

    assert _config_digest(tmp_path) != first
