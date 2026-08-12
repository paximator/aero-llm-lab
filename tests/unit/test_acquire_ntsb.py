import json
from datetime import date
from pathlib import Path

import pytest

from aerollm.data import acquire_ntsb
from aerollm.data.sources import RemoteResponse


def test_acquisition_snapshots_raw_response_and_prints_provider_neutral_record(
    tmp_path, monkeypatch
) -> None:
    artifact_root = tmp_path / "snapshots"
    config = tmp_path / "ntsb.toml"
    config.write_text(
        f'''source = "ntsb"
[remote]
base_url_env = "TEST_NTSB_URL"
api_key_env = "TEST_NTSB_KEY"
timeout_seconds = 5
[snapshot]
root = "{artifact_root.as_posix()}"
retain_response_headers = ["content-type", "etag"]
''',
        encoding="utf-8",
    )
    monkeypatch.setenv("TEST_NTSB_URL", "https://api.example.test")
    monkeypatch.setenv("TEST_NTSB_KEY", "credential-not-for-artifacts")

    def fake_fetch(self, report):
        return RemoteResponse(
            b'{"unchanged": true}',
            "application/json",
            {"ETag": "version-1", "X-Api-Key": "credential-not-for-artifacts"},
        )

    monkeypatch.setattr(acquire_ntsb.NTSBSource, "fetch_report", fake_fetch)
    result = acquire_ntsb.acquire(date(2026, 1, 1), date(2026, 1, 2), config)

    body_path = Path(result["artifact_path"])
    assert body_path.read_bytes() == b'{"unchanged": true}'
    sidecar = json.loads(body_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["request_parameters"] == {
        "endDate": "2026-01-02",
        "startDate": "2026-01-01",
    }
    assert "credential-not-for-artifacts" not in json.dumps(sidecar)
    assert set(result) >= {"source", "source_id", "sha256", "artifact_path"}


def test_acquisition_rejects_more_than_31_days_before_network(tmp_path) -> None:
    with pytest.raises(ValueError, match="31"):
        acquire_ntsb.acquire(date(2026, 1, 1), date(2026, 2, 1), tmp_path / "absent.toml")


def test_command_reports_missing_environment_without_credential_value(
    tmp_path, monkeypatch, capsys
) -> None:
    config = tmp_path / "ntsb.toml"
    config.write_text(
        """[remote]
base_url_env = "ABSENT_NTSB_URL"
api_key_env = "ABSENT_NTSB_KEY"
[snapshot]
root = "ignored"
""",
        encoding="utf-8",
    )
    monkeypatch.delenv("ABSENT_NTSB_URL", raising=False)
    result = acquire_ntsb.main(
        ["--start", "2026-01-01", "--end", "2026-01-01", "--config", str(config)]
    )
    assert result == 2
    assert "ABSENT_NTSB_URL" in capsys.readouterr().err
