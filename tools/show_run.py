"""
Read back what a pipeline run actually did.

    python -m tools.show_run                    index every record
    python -m tools.show_run --failed           only runs that did not accept
    python -m tools.show_run block_hole         full detail for one part
    python -m tools.show_run block_hole --raw   the backend's stdout instead

The raw log is the least informative artifact a run leaves behind. The RECORD
carries the per-iteration evidence and the action taken with its rationale, which
is what explains why a part did not accept. This renders the record; --raw is
there for the cases where the backend itself printed something.

Reads both artifact shapes: tools.run_pipeline writes a LIST of records, and
pipeline.export writes a single record object.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

# `show_run | head` closes the pipe early; without this Python raises
# BrokenPipeError and prints a traceback over the output the user wanted.
if hasattr(signal, "SIGPIPE"):
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

DIM, B, R = "\033[2m", "\033[1m", "\033[0m"
GRN, YEL, RED, CYN = "\033[32m", "\033[33m", "\033[31m", "\033[36m"
if not sys.stdout.isatty():
    DIM = B = R = GRN = YEL = RED = CYN = ""

GOOD = {"accepted"}
SOFT = {"score_plateau", "budget_exhausted", "escalated"}


def tint(outcome: str, width: int = 0) -> str:
    """Colour an outcome, padding BEFORE the escape codes.

    Padding the coloured string instead counts the ANSI bytes toward the field
    width, so every column after it shifts on a terminal and not in a pipe.
    """
    text = f"{outcome:<{width}}" if width else outcome
    if outcome in GOOD:
        return f"{GRN}{text}{R}"
    if outcome in SOFT:
        return f"{YEL}{text}{R}"
    return f"{RED}{text}{R}"


def load_records(d: Path):
    """Yield (path, record) for every run record under d, either shape."""
    seen = set()
    for jf in sorted(d.glob("*.json")):
        try:
            data = json.loads(jf.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        for rec in (data if isinstance(data, list) else [data]):
            if not isinstance(rec, dict) or "outcome" not in rec:
                continue          # a Stage 0 report or a manifest, not a run
            key = (rec.get("model"), rec.get("outcome"), len(rec.get("iterations", [])))
            if key in seen:
                continue          # the same run exported twice
            seen.add(key)
            yield jf, rec


def stem_of(rec: dict, jf: Path) -> str:
    model = rec.get("model") or jf.stem
    return Path(model).stem


def best_of(rec: dict):
    b = rec.get("best_index")
    if b is None:
        return None
    return next((i for i in rec.get("iterations", []) if i.get("index") == b), None)


# ---------------------------------------------------------------------------

def show_index(d: Path, only_failed: bool) -> int:
    rows = []
    for jf, rec in load_records(d):
        oc = rec.get("outcome", "?")
        if only_failed and oc in GOOD:
            continue
        best = best_of(rec)
        last = (rec.get("iterations") or [{}])[-1]
        rows.append({
            "stem": stem_of(rec, jf), "outcome": oc,
            "iters": rec.get("n_iterations", len(rec.get("iterations", []))),
            "tris": (best or last).get("n_triangles", 0),
            "score": (best or {}).get("score", 0.0),
            "sec": rec.get("elapsed_seconds", 0.0),
            "valid": best is not None,
        })
    if not rows:
        print(f"no run records in {d}/"
              + ("" if not only_failed else " matching --failed"))
        return 1

    rows.sort(key=lambda r: (r["outcome"] in GOOD, -r["sec"]))
    print(f"\n{B}{'part':<40}{'outcome':<22}{'it':>3}{'tris':>9}"
          f"{'score':>7}{'sec':>8}{R}")
    print("-" * 89)
    for r in rows:
        mark = "" if r["valid"] else "*"
        print(f"{r['stem'][:39]:<40}{tint(r['outcome'], 22)}"
              f"{r['iters']:>3}{r['tris']:>8,}{mark:<1}"
              f"{r['score']:>7.3f}{r['sec']:>8.1f}")
    n_ok = sum(1 for r in rows if r["outcome"] in GOOD)
    print(f"\n{len(rows)} run(s), {n_ok} accepted."
          f"  {DIM}* = no valid mesh; count is the last attempt{R}")
    print(f"{DIM}detail: python -m tools.show_run <part>{R}")
    return 0


def show_detail(d: Path, name: str, raw: bool) -> int:
    name = Path(name).stem
    hits = [(jf, rec) for jf, rec in load_records(d) if stem_of(rec, jf) == name]
    if not hits:
        near = sorted({stem_of(jf_rec[1], jf_rec[0]) for jf_rec in load_records(d)
                       if name.lower() in stem_of(jf_rec[1], jf_rec[0]).lower()})
        print(f"no record for '{name}' in {d}/")
        if near:
            print("did you mean: " + ", ".join(near[:6]))
        return 1
    jf, rec = hits[0]

    if raw:
        log = d / f"{name}.log"
        if not log.exists():
            print(f"no raw log at {log}")
            return 1
        print(log.read_text())
        return 0

    print(f"\n{B}{name}{R}   {tint(rec.get('outcome', '?'))}"
          f"   {rec.get('elapsed_seconds', 0):.1f}s"
          f"   {DIM}{jf}{R}")

    crit = rec.get("criteria") or {}
    if crit:
        print(f"{DIM}criteria: min_shape {crit.get('min_shape')}, "
              f"min_angle {crit.get('min_angle_deg')}, "
              f"max_gradation {crit.get('max_gradation')}, "
              f"budget {rec.get('budget'):,}{R}")

    for it in rec.get("iterations", []):
        state = ("READY" if it.get("ready") else
                 "valid" if it.get("is_valid") else "INVALID")
        colour = GRN if it.get("ready") else (YEL if it.get("is_valid") else RED)
        print(f"\n  {B}[{it.get('index')}]{R} {it.get('n_triangles', 0):>8,} tris"
              f"   {colour}{state}{R}   score {it.get('score', 0):.3f}"
              f"   {it.get('seconds', 0):.2f}s"
              + (f"   est {it.get('estimated_elements', 0):,}"
                 if it.get("estimated_elements") else ""))

        if it.get("note"):
            print(f"      {DIM}note: {it['note']}{R}")
        if it.get("ineffective"):
            print(f"      {YEL}the previous action had no measurable effect{R}")

        for f in it.get("failures", []):
            print(f"      {RED}HARD{R} {f['code']}"
                  + (f" x{f['count']}" if f.get("count", 1) > 1 else ""))
            if f.get("message"):
                print(f"           {DIM}{f['message'][:96]}{R}")

        missed = [o for o in it.get("objectives", []) if not o.get("satisfied")]
        for o in missed:
            arrow = ">=" if o.get("value", 0) < o.get("target", 0) else "<="
            print(f"      {YEL}MISS{R} {o['name']:<18}"
                  f"{o.get('value', 0):>10.4g} {arrow} {o.get('target', 0):<10.4g}")

        c = it.get("contract")
        if c and not c.get("satisfied", True):
            for v in c.get("violations", []):
                pids = ", ".join(v.get("pids", [])[:3])
                print(f"      {RED}CONTRACT{R} {v['code']}"
                      + (f": {pids}" if pids else ""))

        lr = it.get("load_report") or {}
        if lr.get("heal"):
            h = lr["heal"]
            print(f"      {CYN}reload{R} healed at tol {h.get('tolerance'):.5g}: "
                  f"faces {h.get('faces_before')} -> {h.get('faces_after')} "
                  f"({h.get('removed')} removed)")

        a = it.get("action")
        if a:
            scope = f" on {len(a.get('scope', []))} face(s)" if a.get("scope") else ""
            print(f"      {CYN}-> {a['kind']}{scope}{R}: {a.get('rationale', '')}")

    best = best_of(rec)
    print()
    if best is not None:
        print(f"  best: iteration {best['index']}, {best['n_triangles']:,} tris, "
              f"score {best['score']:.3f}"
              + (" (READY)" if best.get("ready") else " (valid, not ready)"))
    else:
        print(f"  {RED}best: none — no valid mesh was produced{R}")

    for n in rec.get("notes", []):
        print(f"  {DIM}note: {n[:150]}{R}")

    if (d / f"{name}.log").exists():
        print(f"\n{DIM}raw backend output: "
              f"python -m tools.show_run {name} --raw{R}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Inspect pipeline run records")
    ap.add_argument("name", nargs="?", help="part name; omit to list everything")
    ap.add_argument("--dir", default="out/runs", help="where records live")
    ap.add_argument("--failed", action="store_true",
                    help="index only runs that did not accept")
    ap.add_argument("--raw", action="store_true",
                    help="print the backend's stdout instead of the record")
    args = ap.parse_args(argv)

    d = Path(args.dir)
    if not d.is_dir():
        print(f"no run directory at {d}/ — run ./run.sh mesh first")
        return 1
    return (show_detail(d, args.name, args.raw) if args.name
            else show_index(d, args.failed))


if __name__ == "__main__":
    raise SystemExit(main())
