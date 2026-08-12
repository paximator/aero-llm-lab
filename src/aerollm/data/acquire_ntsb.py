"""Deliberately bounded snapshot-first NTSB acquisition command."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

from aerollm.data.ntsb import NTSBRequestError, NTSBSource
from aerollm.data.snapshots import SnapshotStore

MAX_RANGE_DAYS = 31


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Snapshot one bounded NTSB date-range response")
    parser.add_argument(
        "--start", type=date.fromisoformat, required=True, help="inclusive YYYY-MM-DD"
    )
    parser.add_argument(
        "--end", type=date.fromisoformat, required=True, help="inclusive YYYY-MM-DD"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/data/ntsb.toml"))
    return parser


def acquire(start: date, end: date, config_path: Path) -> dict[str, Any]:
    days = (end - start).days + 1
    if not 1 <= days <= MAX_RANGE_DAYS:
        raise ValueError(f"date range must contain between 1 and {MAX_RANGE_DAYS} days")

    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    remote = config["remote"]
    snapshot = config["snapshot"]
    base_url = _required_environment(remote["base_url_env"])
    api_key = _required_environment(remote["api_key_env"])
    source = NTSBSource(
        base_url=base_url,
        api_key=api_key,
        endpoint_path=remote.get("endpoint_path", "aviation/api/GetCasesByDateRangeV2"),
        timeout_seconds=float(remote.get("timeout_seconds", 30)),
    )
    report = next(source.list_reports(start, end))
    response = source.fetch_report(report)
    store = SnapshotStore(
        Path(snapshot["root"]),
        retained_headers=frozenset(snapshot.get("retain_response_headers", [])),
    )
    document = store.save(
        source=source.name,
        source_id=report.source_id,
        source_url=report.source_url,
        publisher="National Transportation Safety Board",
        response=response,
        request_parameters=report.attributes,
        usage_note="Raw NTSB Enterprise Aviation API date-range response",
    )
    return document.to_dict()


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"required environment variable {name} is not set")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        result = acquire(arguments.start, arguments.end, arguments.config)
    except (KeyError, NTSBRequestError, OSError, ValueError, tomllib.TOMLDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
