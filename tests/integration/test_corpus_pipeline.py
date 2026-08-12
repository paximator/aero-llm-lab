import json
from pathlib import Path

from aerollm.data.corpus import CorpusConfig, build_corpus, write_corpus
from aerollm.data.corpus_cli import main
from tests.unit.test_corpus_builder import _input_report


def test_source_manifest_to_frozen_corpus_cli(tmp_path: Path, capsys) -> None:
    source_manifest, config = _input_report(tmp_path, event_id="ERA24LA001")
    config_path = tmp_path / "corpus.toml"
    config_path.write_text(
        "\n".join(
            (
                "schema_version = 1", 'version = "pilot-v1"', 'seed = "fixed"',
                "train_fraction = 0.8", "development_fraction = 0.1",
                "test_fraction = 0.1", "minimum_chunk_characters = 20",
                f'chunk_root = "{config.chunk_root.as_posix()}"',
                f'output = "{config.output.as_posix()}"',
            )
        ),
        encoding="utf-8",
    )

    assert main([str(source_manifest), "--config", str(config_path)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["documents"] == 1
    assert summary["chunks"] == 1
    assert Path(summary["output"]).exists()

    direct = build_corpus([source_manifest], CorpusConfig.load(config_path))
    direct_path = tmp_path / "direct.json"
    direct_digest, _ = write_corpus(direct, direct_path)
    assert direct_digest == summary["sha256"]
