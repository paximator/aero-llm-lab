import json
from pathlib import Path

from aerollm.data import acquire_ntsb_reference
from aerollm.data.sources import RemoteResponse


def test_reference_command_snapshots_without_persisting_key(tmp_path, monkeypatch) -> None:
    root = tmp_path / "snapshots"
    config = tmp_path / "ntsb.toml"
    config.write_text(
        f'''[remote]
base_url = "https://api.example.test/public/api"
api_key_env = "TEST_NTSB_KEY"
endpoint_path = "Common/v2/GetCasesByDateRange/"
[remote.reference_paths]
version = "getversion"
aviation_data_dictionary = "Aviation/v1/GetAviationDataDictionary"
[snapshot]
root = "{root.as_posix()}"
retain_response_headers = ["content-type"]
''',
        encoding="utf-8",
    )
    monkeypatch.setenv("TEST_NTSB_KEY", "credential-not-for-artifacts")
    monkeypatch.setattr(
        acquire_ntsb_reference.NTSBSource,
        "fetch_reference",
        lambda self, path: RemoteResponse(b'{"version":"test"}', "application/json"),
    )

    result = acquire_ntsb_reference.acquire("version", config)
    body_path = Path(result["artifact_path"])
    sidecar = body_path.with_suffix(".json").read_text(encoding="utf-8")

    assert body_path.read_bytes() == b'{"version":"test"}'
    assert "credential-not-for-artifacts" not in sidecar
    assert json.loads(sidecar)["source_url"].endswith("/getversion")
