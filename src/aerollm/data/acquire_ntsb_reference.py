"""Snapshot immutable NTSB API reference resources."""

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

RESOURCES = ("version", "aviation_data_dictionary")


def acquire(resource: str, config_path: Path) -> dict[str, Any]:
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    remote = config["remote"]
    snapshot = config["snapshot"]
    paths = remote["reference_paths"]
    if resource not in RESOURCES:
        raise ValueError(f"resource must be one of: {', '.join(RESOURCES)}")
    api_key = os.environ.get(remote["api_key_env"])
    if not api_key:
        raise ValueError(f"required environment variable {remote['api_key_env']} is not set")
    source = NTSBSource(
        base_url=remote["base_url"],
        api_key=api_key,
        endpoint_path=remote["endpoint_path"],
        timeout_seconds=float(remote.get("timeout_seconds", 30)),
    )
    response = source.fetch_reference(paths[resource])
    url = f"{source.base_url.rstrip('/')}/{paths[resource].lstrip('/')}"
    document = SnapshotStore(
        Path(snapshot["root"]),
        retained_headers=frozenset(snapshot.get("retain_response_headers", [])),
    ).save(
        source=source.name,
        source_id=f"reference-{resource.replace('_', '-')}",
        source_url=url,
        publisher="National Transportation Safety Board",
        response=response,
        request_parameters={},
        usage_note=f"NTSB API reference resource: {resource}",
    )
    return document.to_dict()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Snapshot an NTSB API reference resource")
    parser.add_argument("resource", choices=RESOURCES)
    parser.add_argument("--config", type=Path, default=Path("configs/data/ntsb.toml"))
    arguments = parser.parse_args(argv)
    try:
        result = acquire(arguments.resource, arguments.config)
    except (KeyError, NTSBRequestError, OSError, ValueError, tomllib.TOMLDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
