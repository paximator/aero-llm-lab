"""Snapshot one NTSB aviation case detail selected from discovery metadata."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from aerollm.data.ntsb import NTSBRequestError, NTSBSource
from aerollm.data.snapshots import SnapshotStore


def acquire(
    ntsb_number: str, mkey: int | None, config_path: Path
) -> dict[str, Any]:
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    remote = config["remote"]
    snapshot = config["snapshot"]
    api_key = os.environ.get(remote["api_key_env"])
    if not api_key:
        raise ValueError(f"required environment variable {remote['api_key_env']} is not set")
    source = NTSBSource(
        base_url=remote["base_url"],
        api_key=api_key,
        endpoint_path=remote["endpoint_path"],
        timeout_seconds=float(remote.get("timeout_seconds", 30)),
    )
    url, response = source.fetch_aviation_case(
        remote["reference_paths"]["aviation_case"],
        ntsb_number=ntsb_number,
        mkey=mkey,
    )
    parameters: dict[str, str | int] = {"ntsbNumber": ntsb_number.strip()}
    if mkey is not None:
        parameters["mkey"] = mkey
    identifier = f"aviation-case-{ntsb_number.strip()}"
    if mkey is not None:
        identifier += f"-{mkey}"
    document = SnapshotStore(
        Path(snapshot["root"]),
        retained_headers=frozenset(snapshot.get("retain_response_headers", [])),
    ).save(
        source=source.name,
        source_id=identifier,
        source_url=url,
        publisher="National Transportation Safety Board",
        response=response,
        request_parameters=parameters,
        event_id=ntsb_number.strip(),
        usage_note="Raw NTSB Aviation v1 case-detail response",
    )
    return document.to_dict()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Snapshot one NTSB aviation case detail")
    parser.add_argument("--ntsb-number", required=True)
    parser.add_argument("--mkey", type=int)
    parser.add_argument("--config", type=Path, default=Path("configs/data/ntsb.toml"))
    arguments = parser.parse_args(argv)
    try:
        result = acquire(arguments.ntsb_number, arguments.mkey, arguments.config)
    except (KeyError, NTSBRequestError, OSError, ValueError, tomllib.TOMLDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
