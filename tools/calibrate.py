"""
Threshold calibration log.

The answer to "how do I keep a log of the thresholds" is: don't keep one by hand.
Every finding already records what it tripped and by how much (Finding.threshold /
.measured), and every report records the full config it ran under. So the reports
ARE the log. This tool aggregates them and fits cut points from your verdicts.

Workflow
--------
1. Run Stage 0 across your corpus, always with --json:

       for f in models/*.step; do
         python -m stage0.run "$f" --json "out/$(basename "$f" .step).stage0.json"
       done

2. Collect every threshold-bearing finding into a labelling sheet:

       python -m tools.calibrate collect out/ --csv calibration.csv

3. Open calibration.csv and fill in the `verdict` column for each row:

       real  -- the finding is correct, this really is a defect
       false -- false positive, the geometry is fine
       (blank) -- undecided, ignored by the fit

4. Ask what threshold your verdicts imply:

       python -m tools.calibrate fit calibration.csv

   For each threshold it reports the measured range of `real` vs `false` rows and
   whether they separate. If they do, it proposes a cut. If they overlap, the
   metric cannot distinguish the two cases at any threshold, and the fix is a
   better metric -- not a better number.

   That last case is the one worth knowing about. `face.sliver` on area was
   exactly it: no area threshold separates a thin rail from a legitimate face,
   because area is not the property that makes a face a sliver. Circularity is.

5. Write the accepted values into thresholds.toml and pass --config.

Re-running step 2 after any change is cheap, so keep the CSV in version control
alongside the TOML. The pair is your audit trail: what you decided, and why.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

FIELDS = ["borderline", "part", "code", "severity", "dim", "tag",
          "threshold_name", "threshold", "measured", "ratio", "verdict", "note"]

BORDERLINE_FACTOR = 2.0
"""A row whose measured/threshold ratio is within this factor of 1.0 is one the
threshold actually decided. Those are the rows worth labelling first: a finding
three orders of magnitude clear of its cut tells you nothing about the cut."""


def collect(report_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for jf in sorted(report_dir.glob("*.json")):
        try:
            rep = json.loads(jf.read_text())
        except json.JSONDecodeError:
            print(f"  skipping unparseable {jf.name}")
            continue
        part = Path(rep.get("source_path", jf.stem)).stem
        for f in rep.get("findings", []):
            if not f.get("threshold_name"):
                continue          # not threshold-driven, nothing to calibrate
            thr, meas = f.get("threshold"), f.get("measured")
            ratio = (meas / thr) if (thr not in (None, 0) and meas is not None) else ""
            border = (ratio != "" and 1.0 / BORDERLINE_FACTOR <= ratio <= BORDERLINE_FACTOR)
            rows.append({
                "borderline": "YES" if border else "",
                "part": part,
                "code": f.get("code", ""),
                "severity": f.get("severity", ""),
                "dim": f.get("dim", ""),
                "tag": f.get("tag", ""),
                "threshold_name": f["threshold_name"],
                "threshold": thr,
                "measured": meas,
                "ratio": f"{ratio:.4g}" if ratio != "" else "",
                "verdict": "",
                "note": "",
            })
    return rows


def merge_verdicts(rows: list[dict], existing_csv: Path) -> list[dict]:
    """Carry forward verdicts already recorded, keyed by part+tag+threshold."""
    if not existing_csv.exists():
        return rows
    prior: dict[tuple, tuple[str, str]] = {}
    with existing_csv.open() as fh:
        for r in csv.DictReader(fh):
            key = (r["part"], r["code"], str(r["tag"]), r["threshold_name"])
            if r.get("verdict"):
                prior[key] = (r["verdict"], r.get("note", ""))
    kept = 0
    for row in rows:
        key = (row["part"], row["code"], str(row["tag"]), row["threshold_name"])
        if key in prior:
            row["verdict"], row["note"] = prior[key]
            kept += 1
    if kept:
        print(f"  carried forward {kept} existing verdict(s)")
    return rows


def cmd_collect(args) -> int:
    rows = collect(Path(args.report_dir))
    if not rows:
        print("no threshold-bearing findings. Did you run Stage 0 with --json?")
        return 1
    out = Path(args.csv)
    rows = merge_verdicts(rows, out)
    # Borderline rows first: they are where labelling changes the outcome.
    rows.sort(key=lambda r: (r["borderline"] != "YES", r["threshold_name"],
                             float(r["ratio"]) if r["ratio"] else 0.0))
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    by_thr: dict[str, int] = {}
    for r in rows:
        by_thr[r["threshold_name"]] = by_thr.get(r["threshold_name"], 0) + 1
    print(f"{len(rows)} finding(s) -> {out}")
    for k, v in sorted(by_thr.items()):
        print(f"  {k:<34} {v}")
    unlabelled = sum(1 for r in rows if not r["verdict"])
    border = [r for r in rows if r["borderline"] == "YES"]
    print(f"\n{unlabelled} row(s) need a verdict (real / false)")
    if border:
        print(f"\n{len(border)} BORDERLINE row(s) -- label these first, the "
              f"threshold decided them:")
        for r in border[:12]:
            print(f"  {r['part']:<16} {r['code']:<16} {r['threshold_name']:<22} "
                  f"ratio {r['ratio']}")
    else:
        print("\nno borderline rows: every finding is well clear of its cut, so "
              "this corpus cannot calibrate the numbers.")
    return 0


def cmd_fit(args) -> int:
    with Path(args.csv).open() as fh:
        rows = list(csv.DictReader(fh))

    groups: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        v = (r.get("verdict") or "").strip().lower()
        if v not in ("real", "false"):
            continue
        try:
            m = float(r["measured"])
        except (TypeError, ValueError):
            continue
        groups.setdefault(r["threshold_name"],
                          {"real": [], "false": []})[v].append(m)

    if not groups:
        print("no labelled rows. Fill the `verdict` column with real / false.")
        return 1

    for name, g in sorted(groups.items()):
        real, false = sorted(g["real"]), sorted(g["false"])
        print(f"\n=== {name} ===")
        print(f"  real  n={len(real):<3}" +
              (f" range [{real[0]:.4g}, {real[-1]:.4g}]" if real else " (none)"))
        print(f"  false n={len(false):<3}" +
              (f" range [{false[0]:.4g}, {false[-1]:.4g}]" if false else " (none)"))

        if not real or not false:
            print("  need both labels to fit a cut")
            continue

        # A finding fires when measured < threshold, so `real` values should sit
        # BELOW `false` values. Separable iff max(real) < min(false).
        if max(real) < min(false):
            lo, hi = max(real), min(false)
            proposed = statistics.geometric_mean([lo, hi]) if lo > 0 else hi / 2
            print(f"  SEPARABLE: real all below {hi:.4g}, false all above {lo:.4g}")
            print(f"  proposed threshold: {proposed:.4g}  "
                  f"(geometric midpoint of [{lo:.4g}, {hi:.4g}])")
        else:
            overlap_lo = max(min(real), min(false))
            overlap_hi = min(max(real), max(false))
            n_over = sum(1 for x in real + false if overlap_lo <= x <= overlap_hi)
            print(f"  OVERLAPPING in [{overlap_lo:.4g}, {overlap_hi:.4g}] "
                  f"({n_over} rows)")
            print("  No threshold on this metric separates real from false.")
            print("  The metric is wrong, not the number -- change what you measure.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 0 threshold calibration log")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="aggregate reports into a labelling CSV")
    c.add_argument("report_dir", nargs="?", default="out")
    c.add_argument("--csv", default="calibration.csv")
    c.set_defaults(func=cmd_collect)

    f = sub.add_parser("fit", help="propose thresholds from labelled verdicts")
    f.add_argument("csv", nargs="?", default="calibration.csv")
    f.set_defaults(func=cmd_fit)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
