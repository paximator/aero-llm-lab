"""CLI for planning and materializing the bounded NTSB pilot corpus."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections.abc import Sequence
from dataclasses import replace
from datetime import date
from pathlib import Path

from aerollm.data.corpus import CorpusConfig, build_corpus, write_corpus
from aerollm.data.ntsb import NTSBRequestError
from aerollm.data.pdf import PDFParseError
from aerollm.data.pilot import PilotConfig, PilotPlan, discover_plan, materialize_plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan or build a diverse NTSB pilot corpus")
    parser.add_argument("--start-date", type=date.fromisoformat, required=True)
    parser.add_argument("--end-date", type=date.fromisoformat, required=True)
    parser.add_argument("--target-reports", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--refresh-plan", action="store_true")
    parser.add_argument("--pilot-config", type=Path, default=Path("configs/data/pilot.toml"))
    parser.add_argument("--ntsb-config", type=Path, default=Path("configs/data/ntsb.toml"))
    parser.add_argument("--chunk-config", type=Path, default=Path("configs/data/chunking.toml"))
    parser.add_argument("--corpus-config", type=Path, default=Path("configs/data/corpus.toml"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        pilot = PilotConfig.load(arguments.pilot_config)
        if arguments.target_reports is not None:
            pilot = replace(pilot, target_reports=arguments.target_reports)
        if pilot.plan_path.exists() and not arguments.refresh_plan:
            plan = PilotPlan.load(pilot.plan_path)
            if (plan.start, plan.end, len(plan.candidates)) != (
                arguments.start_date, arguments.end_date, pilot.target_reports,
            ):
                raise ValueError("existing plan does not match request; pass --refresh-plan")
        else:
            plan = discover_plan(
                arguments.start_date, arguments.end_date,
                pilot=pilot, ntsb_config_path=arguments.ntsb_config,
            )
        if arguments.dry_run:
            print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
            return 0
        manifests = materialize_plan(
            plan, pilot=pilot, ntsb_config_path=arguments.ntsb_config,
            chunk_config_path=arguments.chunk_config,
        )
        if len(manifests) != len(plan.candidates):
            raise ValueError(
                f"{len(plan.candidates) - len(manifests)} reports failed; "
                f"see {pilot.failure_report} and rerun to resume"
            )
        corpus_config = CorpusConfig.load(arguments.corpus_config)
        result = build_corpus(manifests, corpus_config)
        digest, build_path = write_corpus(result, corpus_config.output)
    except (
        KeyError, NTSBRequestError, OSError, PDFParseError, TypeError, ValueError,
        json.JSONDecodeError, tomllib.TOMLDecodeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "corpus": corpus_config.output.as_posix(),
                "build_manifest": build_path.as_posix(), "sha256": digest,
                "reports": len(result.corpus.documents), "chunks": len(result.corpus.chunks),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

