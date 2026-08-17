"""
Run record. PURE -- no gmsh, no backend.

Every iteration's inputs, evidence, and decision, in a serializable form.

This is not logging. The loop is only as trustworthy as its scoring function, and
the record is what lets thresholds be calibrated against real parts rather than
guessed -- the same argument that made tools/calibrate.py worth building for Stage
0. It is also the artifact a future agent would read to understand why a run ended
where it did.

Three things each iteration must capture, or the record cannot explain itself:
  * the exact parameters meshed with (fingerprint + spec diff from the previous)
  * the verdict, split into hard failures and objective attainment
  * the action chosen, its rationale, and the evidence behind it
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum


class Outcome(str, Enum):
    ACCEPTED = "accepted"
    BUDGET_EXHAUSTED = "budget_exhausted"
    MAX_ITERATIONS = "max_iterations"
    NO_ACTION = "no_action_available"
    OSCILLATION = "oscillation_detected"
    PLATEAU = "score_plateau"
    ESCALATED = "escalated"
    BACKEND_ERROR = "backend_error"

    @property
    def succeeded(self) -> bool:
        return self is Outcome.ACCEPTED


@dataclass
class Iteration:
    index: int
    request_fingerprint: str = ""
    size_field: dict = field(default_factory=dict)
    spec_diff: list = field(default_factory=list)
    strategy: dict = field(default_factory=dict)
    estimated_elements: int = 0
    n_triangles: int = 0
    n_vertices: int = 0
    seconds: float = 0.0
    is_valid: bool = False
    ready: bool = False
    score: float = 0.0
    failures: list = field(default_factory=list)
    objectives: list = field(default_factory=list)
    worst_faces: list = field(default_factory=list)
    action: dict | None = None
    contract: dict | None = None
    """Topology and feature contract verification for this iteration."""
    load_report: dict | None = None
    """Backend load report when this iteration followed a geometry-changing
    strategy action. Without it there is no way to tell 'healing ran and did not
    help' from 'healing silently did nothing' -- sliver_block produced byte-identical
    meshes across an enable_healing action and the record could not explain why."""
    over_budget: bool = False
    """Set when the size field was refused before meshing. Distinct from
    NO_ACTION: the policy had a move, the budget forbade it."""
    ineffective: bool = False
    """Set when a strategy action produced an indistinguishable mesh."""
    note: str = ""

    def summary(self) -> str:
        state = ("READY" if self.ready else
                 "valid" if self.is_valid else "INVALID")
        head = (f"[{self.index}] {self.n_triangles:>7,} tris  {state:<7} "
                f"score {self.score:.3f}  {self.seconds:.2f}s")
        if not self.is_valid:
            head += "  " + ", ".join(f["code"] for f in self.failures[:3])
        else:
            missed = [o["name"] for o in self.objectives if not o["satisfied"]]
            if missed:
                head += "  miss: " + ", ".join(missed[:4])
        if self.contract and not self.contract.get("satisfied", True):
            # Name the entities. An escalation that says a decision is needed but
            # not WHAT to decide on forces a JSON dig for the one fact required to
            # act, which is the difference between a report and a prompt.
            for v in self.contract["violations"][:3]:
                pids = v.get("pids") or []
                shown = ", ".join(pids[:4]) + ("..." if len(pids) > 4 else "")
                head += f"\n      contract [{v['code']}]"
                head += f": {shown}" if shown else ""
        if self.ineffective:
            head += "  [previous action had NO EFFECT]"
        if self.load_report:
            head += f"\n      reload: {self.load_report}"
        if self.action:
            head += f"\n      -> {self.action['kind']}: {self.action['rationale']}"
        return head


@dataclass
class RunRecord:
    model: str = ""
    backend: str = ""
    selector: str = ""
    criteria: dict = field(default_factory=dict)
    budget: int = 0
    started: float = field(default_factory=time.time)
    iterations: list[Iteration] = field(default_factory=list)
    outcome: Outcome = Outcome.MAX_ITERATIONS
    best_index: int | None = None
    """Index of the best VALID iteration -- not necessarily the last one.

    The loop can degrade: the final iteration is often the most aggressive edit and
    may be worse than one before it, or invalid. Returning the last mesh would throw
    away a good result the run already found.
    """
    notes: list[str] = field(default_factory=list)

    # -- accounting ----------------------------------------------------------
    def add(self, it: Iteration) -> None:
        self.iterations.append(it)
        if it.is_valid:
            cur = self.best
            if cur is None or (it.ready, it.score) > (cur.ready, cur.score):
                self.best_index = it.index

    @property
    def best(self) -> Iteration | None:
        if self.best_index is None:
            return None
        for it in self.iterations:
            if it.index == self.best_index:
                return it
        return None

    @property
    def elapsed(self) -> float:
        return time.time() - self.started

    def score_history(self) -> list[float]:
        return [it.score for it in self.iterations]

    def plateaued(self, window: int = 2, eps: float = 1e-3) -> bool:
        """True when the last `window` iterations gained less than eps in total.

        Distinct from oscillation: a plateau means the policy has a lever but the
        lever stopped working, so continuing burns budget for nothing.
        """
        s = self.score_history()
        if len(s) < window + 1:
            return False
        return (max(s[-window:]) - s[-(window + 1)]) < eps

    # -- serialization -------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "model": self.model, "backend": self.backend,
            "selector": self.selector, "criteria": self.criteria,
            "budget": self.budget, "outcome": self.outcome.value,
            "succeeded": self.outcome.succeeded,
            "best_index": self.best_index,
            "n_iterations": len(self.iterations),
            "elapsed_seconds": round(self.elapsed, 3),
            "notes": list(self.notes),
            "iterations": [asdict(it) for it in self.iterations],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def summary(self) -> str:
        lines = [f"{self.model}  [{self.backend}/{self.selector}]  "
                 f"budget {self.budget:,}"]
        lines.extend(it.summary() for it in self.iterations)
        best = self.best
        lines.append(f"outcome: {self.outcome.value}")
        if best is not None:
            lines.append(f"best: iteration {best.index}, {best.n_triangles:,} tris, "
                         f"score {best.score:.3f}"
                         f"{' (READY)' if best.ready else ' (valid, not ready)'}")
        else:
            lines.append("best: none -- no valid mesh was produced")
        for n in self.notes:
            lines.append(f"note: {n}")
        return "\n".join(lines)
