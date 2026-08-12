import base64
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from aerollm.data.ntsb_reports import NTSBReportSource
from aerollm.data.pilot import (
    PilotCandidate,
    PilotConfig,
    PilotPlan,
    _cached_snapshot,
    _candidates_from_snapshot,
    discovery_windows,
    materialize_plan,
    select_diverse,
)
from aerollm.data.snapshots import SnapshotStore
from aerollm.data.sources import RemoteResponse

PDF_FIXTURE = Path(__file__).parents[1] / "fixtures" / "ntsb" / "synthetic-report.pdf.base64"


def test_discovery_windows_are_bounded_and_span_requested_dates() -> None:
    windows = discovery_windows(
        date(2018, 1, 1), date(2025, 12, 31), requests=96, window_days=31
    )

    assert len(windows) == 96
    assert windows[0][0] == date(2018, 1, 1)
    assert windows[-1][0] == date(2025, 12, 31)
    assert all(1 <= (end - start).days + 1 <= 31 for start, end in windows)
    pairs = zip(windows, windows[1:], strict=False)
    assert all(right[0] <= left[1] + timedelta(days=1) for left, right in pairs)


def test_diverse_selection_is_stable_and_prefers_unseen_features() -> None:
    candidates = (
        _candidate("A", 2020, severity="Fatal", weather="IMC"),
        _candidate("B", 2020, severity="Fatal", weather="IMC"),
        _candidate("C", 2021, severity="None", weather="VMC"),
    )

    first = select_diverse(candidates, target=2, seed="fixed")
    second = select_diverse(reversed(candidates), target=2, seed="fixed")

    assert first == second
    assert {item.source_id for item in first} != {"A", "B"}


def test_pilot_plan_round_trip(tmp_path: Path) -> None:
    plan = PilotPlan(
        date(2020, 1, 1), date(2021, 1, 1), "fixed",
        (_candidate("A", 2020, severity="Fatal", weather="IMC"),),
    )
    path = tmp_path / "plan.json"

    plan.write(path)

    assert PilotPlan.load(path) == plan


def test_candidate_extraction_rejects_non_aviation_cases(tmp_path: Path) -> None:
    payload = {
        "data": [
            {
                "ntsbNumber": "DCA24FM001", "eventDate": "2024-01-01",
                "mode": "Marine", "reportNumber": "MIR2401",
            },
            {
                "ntsbNumber": "CEN24FA001", "eventDate": "2024-01-02",
                "mode": "Aviation", "reportNumber": "AAR2401",
            },
        ]
    }
    snapshot = SnapshotStore(tmp_path).save(
        source="ntsb", source_id="cases", source_url="https://api.test/cases",
        publisher="NTSB",
        response=RemoteResponse(json.dumps(payload).encode(), "application/json"),
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    candidates = _candidates_from_snapshot(snapshot)

    assert [candidate.source_id for candidate in candidates] == ["CEN24FA001"]


def test_materialization_is_resumable_and_writes_failure_report(tmp_path: Path) -> None:
    discovery_store = SnapshotStore(tmp_path / "snapshots")
    discovery = discovery_store.save(
        source="ntsb", source_id="cases-window", source_url="https://api.test/cases",
        publisher="NTSB", response=RemoteResponse(b'{"data": []}', "application/json"),
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    candidate = PilotCandidate(
        "DCA20MA059", "DCA20MA059", date(2020, 1, 26),
        "https://www.ntsb.gov/report.pdf",
        str(Path(discovery.artifact_path).with_suffix(".json")),
        {"severity": "Fatal", "weather": "IMC"},
    )
    plan = PilotPlan(date(2020, 1, 1), date(2020, 1, 31), "fixed", (candidate,))
    pilot = PilotConfig(
        "fixed", 1, 1, 31, tmp_path / "plan.json", tmp_path / "manifests",
        tmp_path / "chunks", tmp_path / "failures.json",
    )
    ntsb_config = tmp_path / "ntsb.toml"
    ntsb_config.write_text(
        "\n".join(
            (
                '[snapshot]', f'root = "{(tmp_path / "snapshots").as_posix()}"',
                'retain_response_headers = ["content-type"]', '[reports]',
                'allowed_hosts = ["www.ntsb.gov"]', "timeout_seconds = 5",
                "max_bytes = 1000000", f'parsed_root = "{(tmp_path / "parsed").as_posix()}"',
            )
        ),
        encoding="utf-8",
    )
    chunk_config = tmp_path / "chunking.toml"
    chunk_config.write_text(
        "\n".join(
            (
                "schema_version = 1", "target_characters = 1600",
                "overlap_characters = 256", "minimum_characters = 320",
                "boundary_search_characters = 400",
            )
        ),
        encoding="utf-8",
    )
    calls = 0

    def transport(url, headers, timeout, max_bytes, allowed_hosts):
        nonlocal calls
        calls += 1
        body = base64.b64decode(PDF_FIXTURE.read_text(encoding="ascii"))
        return RemoteResponse(body, "application/pdf")

    source = NTSBReportSource(max_bytes=1_000_000, transport=transport)
    first = materialize_plan(
        plan, pilot=pilot, ntsb_config_path=ntsb_config,
        chunk_config_path=chunk_config, report_source=source,
        now=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )
    second = materialize_plan(
        plan, pilot=pilot, ntsb_config_path=ntsb_config,
        chunk_config_path=chunk_config, report_source=source,
        now=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert first == second
    assert calls == 1
    assert json.loads(pilot.failure_report.read_text())["failed"] == []
    assert first[0].exists()


def test_pilot_configuration_rejects_unbounded_requests(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot exceed 100"):
        PilotConfig(
            "fixed", 1, 101, 31, tmp_path / "p", tmp_path / "m",
            tmp_path / "c", tmp_path / "f",
        )


def test_cached_snapshot_requires_matching_request_parameters(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    saved = store.save(
        source="ntsb", source_id="cases-window", source_url="https://api.test/cases",
        publisher="NTSB", response=RemoteResponse(b'{"data": []}', "application/json"),
        request_parameters={
            "startDate": "2020-01-01", "endDate": "2020-01-31", "mode": "Aviation"
        },
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    found = _cached_snapshot(
        tmp_path, "cases-window",
        {"startDate": "2020-01-01", "endDate": "2020-01-31", "mode": "Aviation"},
    )
    missing = _cached_snapshot(
        tmp_path, "cases-window",
        {"startDate": "2020-01-01", "endDate": "2020-01-31"},
    )

    assert found == saved
    assert missing is None


def _candidate(source_id: str, year: int, **attributes: str) -> PilotCandidate:
    return PilotCandidate(
        source_id, source_id, date(year, 1, 1), f"https://ntsb.test/{source_id}.pdf",
        "snapshot.json", attributes,
    )
