"""
Run the pipeline over a corpus: resumable, parallel, per-file timeout, aggregated.

    python -m tools.corpus models/abc                    # run what is not done
    python -m tools.corpus models/abc --only score_plateau
    python -m tools.corpus models/abc --force -j 8
    python -m tools.corpus models/abc --report           # aggregate, run nothing

Replaces the shell loop plus three ad-hoc aggregation one-liners.

Why resume matters
------------------
A 50-part corpus with a 300 s ceiling is a 20-minute round trip, and most of that
is re-proving parts that already passed. After a fix only the unresolved parts are
interesting, so by default anything already `accepted` is skipped. That turns the
iteration loop from twenty minutes into about one.

Why a separate process per file
-------------------------------
Per-file timeouts. A single invocation over a glob lets one pathological part
consume the whole budget, and the ABC corpus has parts that stall gmsh for minutes
before producing anything. A process per file also contains OCC crashes, which take
the interpreter with them rather than raising.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# `corpus --report | head` closes the pipe early; without this Python raises
# BrokenPipeError and prints a traceback over the output that was wanted.
if hasattr(signal, "SIGPIPE"):
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

CAD_GLOBS = ("*.step", "*.stp", "*.STEP", "*.STP", "*.iges", "*.igs", "*.brep")

TIMEOUT_RC = 124
"""Reported for a part killed by the timeout, matching GNU timeout."""


def discover(root: Path) -> list[Path]:
    files: list[Path] = []
    for g in CAD_GLOBS:
        files.extend(root.glob(g))
    return sorted(set(files))


def record_path(outdir: Path, stem: str) -> Path:
    return outdir / f"{stem}.run.json"


def load_outcome(path: Path) -> str | None:
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    recs = data if isinstance(data, list) else [data]
    return recs[0].get("outcome") if recs else None


def run_one(cad: Path, outdir: Path, meshdir: Path, *, timeout: int,
            budget: int, max_iterations: int, fidelity: bool) -> dict:
    stem = cad.stem
    cmd = [sys.executable, "-m", "tools.run_pipeline", str(cad),
           "--out", str(meshdir), "--json", str(record_path(outdir, stem)),
           "--budget", str(budget), "--max-iterations", str(max_iterations), "-q"]
    if not fidelity:
        cmd.append("--no-fidelity")

    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        rc, tail = p.returncode, (p.stdout or p.stderr or "").strip().splitlines()
    except subprocess.TimeoutExpired:
        return {"stem": stem, "rc": TIMEOUT_RC, "seconds": time.time() - t0,
                "outcome": "timeout", "note": f"killed after {timeout}s"}
    except Exception as e:                     # a crash must not stop the corpus
        return {"stem": stem, "rc": -1, "seconds": time.time() - t0,
                "outcome": "crash", "note": f"{type(e).__name__}: {e}"}

    return {"stem": stem, "rc": rc, "seconds": time.time() - t0,
            "outcome": load_outcome(record_path(outdir, stem)) or "unknown",
            "note": (tail[-1][:80] if tail else "")}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(outdir: Path) -> dict:
    outcomes, codes, missed = Counter(), Counter(), Counter()
    iters, tris, secs, adapted = [], [], [], 0
    per_part = []

    # Match on SHAPE, not on filename. This module writes "<stem>.run.json" but
    # run.sh's per-file path writes "<stem>.json", so a filename glob silently
    # reported 0 runs after ./run.sh mesh while the meshes sat right there.
    # Checking for an "outcome" key also skips Stage 0 reports and export
    # manifests that share the directory.
    seen = set()
    for jf in sorted(outdir.glob("*.json")):
        try:
            data = json.loads(jf.read_text())
        except Exception:
            continue
        for rec in (data if isinstance(data, list) else [data]):
            if not isinstance(rec, dict) or "outcome" not in rec:
                continue
            key = (rec.get("model"), rec.get("outcome"),
                   len(rec.get("iterations", ())))
            if key in seen:
                continue          # the same run written by both paths
            seen.add(key)
            oc = rec.get("outcome", "unknown")
            outcomes[oc] += 1
            n = rec.get("n_iterations", 0)
            iters.append(n)
            adapted += 1 if n > 1 else 0
            secs.append(rec.get("elapsed_seconds", 0.0))
            for it in rec.get("iterations", []):
                for f in it.get("failures", []):
                    codes[f["code"]] += 1
            b = rec.get("best_index")
            best = None
            if b is not None:
                best = next((i for i in rec["iterations"] if i["index"] == b), None)
                if best:
                    tris.append(best["n_triangles"])
            if oc not in ("accepted", "escalated") and rec.get("iterations"):
                for o in rec["iterations"][-1].get("objectives", []):
                    if not o.get("satisfied"):
                        missed[o["name"]] += 1
            per_part.append({"part": Path(rec.get("model", jf.stem)).stem,
                             "outcome": oc, "iterations": n,
                             "tris": best["n_triangles"] if best else 0,
                             "score": best["score"] if best else 0.0,
                             "seconds": rec.get("elapsed_seconds", 0.0)})

    return {"outcomes": outcomes, "failure_codes": codes, "missed": missed,
            "iterations": iters, "tris": tris, "seconds": secs,
            "adapted": adapted, "per_part": per_part}


def _med(xs):
    return sorted(xs)[len(xs) // 2] if xs else 0


def print_report(agg: dict, meshdir: Path) -> None:
    total = sum(agg["outcomes"].values())
    print(f"\n{'=' * 62}\n{total} run(s)\n{'=' * 62}")
    print("outcomes:")
    for k, v in agg["outcomes"].most_common():
        print(f"  {k:<24}{v:>4}  {100 * v / max(total, 1):>5.1f}%")

    if agg["failure_codes"]:
        print("failure codes:")
        for k, v in agg["failure_codes"].most_common(10):
            print(f"  {k:<34}{v:>4}")

    if agg["missed"]:
        print("objectives unsatisfied at stop:")
        for k, v in agg["missed"].most_common(8):
            print(f"  {k:<24}{v:>4}")

    t, s = agg["tris"], agg["seconds"]
    if t:
        print(f"best-mesh triangles  med/max: {_med(t):,} / {max(t):,}")
    if s:
        print(f"seconds              med/max: {_med(s):.1f} / {max(s):.1f}")
    print(f"adapted (iterations>1): {agg['adapted']} of {len(agg['iterations'])}")
    print(f"meshes exported: {len(list(meshdir.glob('*.msh')))}")

    slow = sorted(agg["per_part"], key=lambda r: -r["seconds"])[:5]
    if slow and slow[0]["seconds"] > 30:
        print("slowest parts:")
        for r in slow:
            print(f"  {r['part'][:44]:<46}{r['seconds']:>7.1f}s  {r['outcome']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the pipeline over a corpus")
    ap.add_argument("root", nargs="?", default="models",
                    help="directory of CAD files (unused with --report)")
    ap.add_argument("--out", default="out/abc", help="run records")
    ap.add_argument("--meshes", default="out/meshes")
    ap.add_argument("-j", "--jobs", type=int, default=4,
                    help="parallel workers. Each is a separate process, so this "
                         "also contains OCC crashes.")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--budget", type=int, default=250_000)
    ap.add_argument("--max-iterations", type=int, default=8)
    ap.add_argument("--fidelity", action="store_true",
                    help="measure chordal/normal deviation; slow (one OCC "
                         "projection per sample)")
    ap.add_argument("--force", action="store_true",
                    help="re-run everything, including parts already accepted")
    ap.add_argument("--only", action="append",
                    help="re-run only parts whose last outcome was this. "
                         "Repeatable, e.g. --only score_plateau --only timeout")
    ap.add_argument("--report", action="store_true",
                    help="aggregate existing records and exit without running")
    args = ap.parse_args(argv)

    root = Path(args.root)
    outdir, meshdir = Path(args.out), Path(args.meshes)
    outdir.mkdir(parents=True, exist_ok=True)
    meshdir.mkdir(parents=True, exist_ok=True)

    if args.report:
        print_report(aggregate(outdir), meshdir)
        return 0

    files = discover(root)
    if not files:
        print(f"no CAD files under {root}")
        return 1

    todo = []
    skipped = 0
    for f in files:
        prev = load_outcome(record_path(outdir, f.stem))
        if args.force:
            todo.append(f)
        elif args.only:
            if prev in set(args.only):
                todo.append(f)
            else:
                skipped += 1
        elif prev == "accepted":
            skipped += 1                # already resolved; nothing to learn
        else:
            todo.append(f)

    print(f"{len(files)} file(s); running {len(todo)}, skipping {skipped} "
          f"(already accepted or filtered). {args.jobs} worker(s), "
          f"{args.timeout}s each.")
    if not todo:
        print_report(aggregate(outdir), meshdir)
        return 0

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futs = [pool.submit(run_one, f, outdir, meshdir, timeout=args.timeout,
                            budget=args.budget,
                            max_iterations=args.max_iterations,
                            fidelity=args.fidelity) for f in todo]
        for fut in futs:
            r = fut.result()
            done += 1
            print(f"[{done:>3}/{len(todo)}] {r['stem'][:42]:<44}"
                  f"{r['outcome']:<20}{r['seconds']:>6.1f}s")

    print(f"\nwall clock: {time.time() - t0:.1f}s")
    print_report(aggregate(outdir), meshdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
