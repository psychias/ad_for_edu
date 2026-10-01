"""The gate in front of every paid call.

A command that would call a hosted model first builds a `SpendEstimate`, prints
it, and asks `require_approval`. Without approval the command stops there with
exit code 0: looking at a price is a successful run, not a failure.

Approval yields a `SpendApproval` token. A model client cannot be constructed
without one, so "no call before approval" is a property of the types and does
not depend on the order of statements in a command.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import TextIO

from .errors import SpendNotApprovedError

APPROVE_FLAG = "--approve-spend"
APPROVE_DEST = "approve_spend"

_MINT = object()


@dataclass(frozen=True)
class SpendLine:
    """One priced row of an estimate: the calls one model makes for one unit of work."""

    row: str
    model: str
    calls: int
    per_call: float
    measured: bool = True
    note: str = ""

    @property
    def total(self) -> float:
        return self.calls * self.per_call


@dataclass
class SpendEstimate:
    """What a stage would cost, line by line, beside what it offers for free."""

    stage: str
    lines: list[SpendLine] = field(default_factory=list)
    free_columns: tuple[str, ...] = ()

    def add(
        self,
        row: str,
        model: str,
        calls: int,
        per_call: float,
        *,
        measured: bool = True,
        note: str = "",
    ) -> None:
        self.lines.append(SpendLine(row, model, int(calls), float(per_call), measured, note))

    @property
    def total(self) -> float:
        return sum(line.total for line in self.lines)

    @property
    def calls(self) -> int:
        return sum(line.calls for line in self.lines)

    @property
    def all_measured(self) -> bool:
        return all(line.measured for line in self.lines)

    def render(self) -> str:
        width = 26
        blank = ""
        out = [f"SPEND ESTIMATE -- stage {self.stage}", ""]
        for column in self.free_columns:
            out.append(
                f"  {column:{width}s} {blank:22s} {blank:>6s} {blank:>8s} "
                f"{'$0.00':>9s}   (local, free)"
            )
        if self.lines:
            out.append(
                f"  {'row':{width}s} {'model':22s} {'calls':>6s} "
                f"{'$/call':>8s} {'total':>9s}"
            )
        for line in self.lines:
            tag = "" if line.measured else "  (token shape assumed, not measured)"
            note = f"  {line.note}" if line.note else ""
            out.append(
                f"  {line.row:{width}s} {line.model:22s} {line.calls:6d} "
                f"${line.per_call:7.4f} ${line.total:8.2f}{tag}{note}"
            )
        out.append(f"  {'TOTAL':{width}s} {'':22s} {self.calls:6d} {'':>8s} ${self.total:8.2f}")
        if self.lines and not self.all_measured:
            out.append(
                "  At least one row rests on an assumed token shape. Run a small pilot and "
                "price the stage from its measured usage before a full run."
            )
        return "\n".join(out)


class SpendApproval:
    """Proof that an estimate was shown and the spend was approved."""

    def __init__(self, mint: object, stage: str, total: float, calls: int) -> None:
        if mint is not _MINT:
            raise SpendNotApprovedError(
                "a SpendApproval is issued by require_approval, after the estimate is shown"
            )
        self.stage = stage
        self.total = total
        self.calls = calls

    def __repr__(self) -> str:
        return f"SpendApproval(stage={self.stage!r}, total={self.total:.2f}, calls={self.calls})"


def require_approval(
    estimate: SpendEstimate,
    approved: bool,
    *,
    out: TextIO | None = None,
) -> SpendApproval:
    """Print the estimate. Return a token when approved; otherwise exit with code 0."""
    stream = out or sys.stdout
    print(estimate.render(), file=stream)
    if not approved:
        print(
            f"\n  no {APPROVE_FLAG}: nothing was called. Re-run with the flag to spend.",
            file=stream,
        )
        raise SystemExit(0)
    print(
        f"\n  {APPROVE_FLAG} given -- proceeding: ${estimate.total:.2f} "
        f"over {estimate.calls} calls.",
        file=stream,
    )
    return SpendApproval(_MINT, estimate.stage, estimate.total, estimate.calls)


def free_approval(stage: str) -> SpendApproval:
    """A token for a stage that calls only local models and therefore costs nothing."""
    return SpendApproval(_MINT, stage, 0.0, 0)
