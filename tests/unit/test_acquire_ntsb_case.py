import json
from pathlib import Path

from aerollm.data import acquire_ntsb_case
from aerollm.data.sources import RemoteResponse


def test_case_command_snapshots_identifiers_without_persisting_key(tmp_path, monkeypatch) -> None:
    root = tmp_path / "snapshots"
    config = tmp_path / "ntsb.toml"
    config.write_text(
        f'''[remote]
base_url = "https://api.example.test/public/api"
api_key_env = "TEST_NTSB_KEY"
endpoint_path = "Common/v2/GetCasesByDateRange/"
[remote.reference_paths]
aviation_case = "Aviation/v1/GetAviationCase/"
[snapshot]
root = "{root.as_posix()}"
retain_response_headers = ["content-type"]
''',
        encoding="utf-8",
    )
    monkeypatch.setenv("TEST_NTSB_KEY", "credential-not-for-artifacts")
    monkeypatch.setattr(
        acquire_ntsb_case.NTSBSource,
        "fetch_aviation_case",
        lambda self, path, **kwargs: (
            "https://api.example.test/public/api/Aviation/v1/GetAviationCase/"
            "?ntsbNumber=WPR26LA075&mkey=202254",
            RemoteResponse(b'{"ntsbNumber":"WPR26LA075"}', "application/json"),
        ),
    )

    result = acquire_ntsb_case.acquire("WPR26LA075", 202254, config)
    body_path = Path(result["artifact_path"])
    sidecar = body_path.with_suffix(".json").read_text(encoding="utf-8")

    assert json.loads(body_path.read_bytes())["ntsbNumber"] == "WPR26LA075"
    assert json.loads(sidecar)["request_parameters"] == {
        "mkey": 202254,
        "ntsbNumber": "WPR26LA075",
    }
    assert "credential-not-for-artifacts" not in sidecar
