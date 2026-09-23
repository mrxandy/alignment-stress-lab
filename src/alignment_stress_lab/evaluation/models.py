"""Typed, provenance-carrying outcomes returned by evaluators."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Verdict(StrEnum):
    """A response-level evaluation outcome.

    INVALID is distinct from COMPLIANCE so empty, truncated, or degenerate
    generations can never count as a successful response.
    """

    REFUSAL = "refusal"
    COMPLIANCE = "compliance"
    INVALID = "invalid"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class JudgeLabel:
    verdict: Verdict
    evaluator: str
    protocol_version: str
    reason: str | None = None
