"""Deterministic event-family split assignment."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping

from aerollm.common.schemas import Split


def assign_family_splits(
    families: Iterable[str],
    *,
    seed: str,
    fractions: Mapping[Split, float],
) -> dict[str, Split]:
    """Assign whole event families using stable ordering and largest-remainder quotas."""
    unique = set(families)
    if not unique:
        raise ValueError("at least one event family is required")
    if not seed:
        raise ValueError("split seed is required")
    if set(fractions) != set(Split) or any(value < 0 for value in fractions.values()):
        raise ValueError("fractions must define non-negative train, development, and test values")
    if not math.isclose(sum(fractions.values()), 1.0, abs_tol=1e-9):
        raise ValueError("split fractions must sum to 1")

    ordered = sorted(unique, key=lambda family: (_digest(seed, family), family))
    quotas = _largest_remainder_quotas(len(ordered), fractions)
    result: dict[str, Split] = {}
    cursor = 0
    for split in Split:
        end = cursor + quotas[split]
        result.update((family, split) for family in ordered[cursor:end])
        cursor = end
    return result


def _largest_remainder_quotas(
    count: int, fractions: Mapping[Split, float]
) -> dict[Split, int]:
    exact = {split: count * fractions[split] for split in Split}
    quotas = {split: math.floor(exact[split]) for split in Split}
    remaining = count - sum(quotas.values())
    priority = {Split.TRAIN: 0, Split.DEVELOPMENT: 1, Split.TEST: 2}
    ranked = sorted(Split, key=lambda split: (-(exact[split] - quotas[split]), priority[split]))
    for split in ranked[:remaining]:
        quotas[split] += 1
    return quotas


def _digest(seed: str, family: str) -> str:
    return hashlib.sha256(f"{seed}\0{family}".encode()).hexdigest()

