import json
from pathlib import Path

import pytest

from aerollm.training.qlora import (
    load_config,
    load_training_messages,
    render_training_chat,
    render_training_prompt,
    tree_sha256,
)


def test_training_config_is_valid() -> None:
    config = load_config(Path("configs/training/local_8gb_qlora.toml"))

    assert config.rank == 16
    assert config.max_sequence_length == 1024
    assert config.optimizer == "paged_adamw_8bit"
    assert "q_proj" in config.target_modules


def test_training_messages_keep_only_chat_payload(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    messages = [{"role": "system", "content": "a"}] * 3
    path.write_text(json.dumps({"records": [{"messages": messages}]}), encoding="utf-8")

    assert load_training_messages(path, 1) == [messages]


def test_training_record_limit_is_validated(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps({"records": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="record limit"):
        load_training_messages(path, 1)


def test_adapter_tree_digest_ignores_its_manifest(tmp_path: Path) -> None:
    (tmp_path / "adapter.bin").write_bytes(b"adapter")
    before = tree_sha256(tmp_path)
    (tmp_path / "run_manifest.json").write_text("{}", encoding="utf-8")

    assert tree_sha256(tmp_path) == before


def test_base_model_chat_rendering_is_explicit() -> None:
    messages = [
        {"role": "system", "content": "Ground answers."},
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": '{"answer":"Fact"}'},
    ]

    assert render_training_chat(messages) == (
        '[SYSTEM]\nGround answers.\n\n[USER]\nQuestion\n\n[ASSISTANT]\n{"answer":"Fact"}'
    )
    assert render_training_prompt(messages).endswith("[ASSISTANT]\n")
