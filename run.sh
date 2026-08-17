#!/usr/bin/env bash
#
# run.sh — single entry point for the CADtoMesh_Agent pipeline.
#
#   ./run.sh help                       what you can do
#   ./run.sh setup                      venv + dependencies
#   ./run.sh test                       300 tests, no gmsh needed
#   ./run.sh all                        parts -> stage0 -> mesh -> report
#   ./run.sh mesh models/block_hole.step
#   ./run.sh corpus models/abc -j 6
#
# Written for bash 3.2 (the version macOS ships) — no associative arrays, no
# ${var,,}, no mapfile. Runs the same under zsh since it sets its own shebang.

set -uo pipefail
# Globs that match nothing must vanish, not survive as literal text. This has to
# be global: a glob is expanded by the CALLER, so enabling nullglob inside a
# function is already too late and the unmatched pattern arrives as an argument.
shopt -s nullglob

# --------------------------------------------------------------- environment
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  B=$'\033[1m'; DIM=$'\033[2m'; R=$'\033[0m'
  GRN=$'\033[32m'; YEL=$'\033[33m'; RED=$'\033[31m'; CYN=$'\033[36m'
else
  B=""; DIM=""; R=""; GRN=""; YEL=""; RED=""; CYN=""
fi

say()  { printf '%s\n' "$*"; }
head_() { printf '\n%s%s%s\n' "$B" "$*" "$R"; }
ok()   { printf '%s✓%s %s\n' "$GRN" "$R" "$*"; }
warn() { printf '%s!%s %s\n' "$YEL" "$R" "$*"; }
die()  { printf '%s✗%s %s\n' "$RED" "$R" "$*" >&2; exit 1; }

# ------------------------------------------------------------------- python
# uv is preferred for creating and populating .venv: it resolves and installs an
# order of magnitude faster, and the venv it produces is an ordinary one, so
# activation below is identical either way.
UV="$(command -v uv 2>/dev/null || true)"

# Activate the project venv if one exists and we are not already inside it.
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  . .venv/bin/activate
fi

# PY_CMD is an ARRAY because `uv run python` is three words. Held as a string it
# would be re-split at every call site, and any site that quoted it would try to
# exec a file literally named "uv run python".
if [ -n "${VIRTUAL_ENV:-}" ]; then
  PY_CMD=("${PYTHON:-python3}")            # inside the venv: plain python
elif [ -n "$UV" ] && [ -f pyproject.toml ]; then
  PY_CMD=(uv run --quiet python)           # uv resolves the project env itself
else
  PY_CMD=("${PYTHON:-python3}")
fi
PY_SHOW="${PY_CMD[*]}"                     # for dry-run and diagnostics

# macOS has no GNU timeout unless coreutils is installed; perl's alarm is
# always present. GNU timeout reports 124 on kill, the perl route 142.
if command -v gtimeout >/dev/null 2>&1;     then _TO() { gtimeout "$@"; }
elif command -v timeout >/dev/null 2>&1;    then _TO() { timeout  "$@"; }
else                                             _TO() { perl -e 'alarm shift; exec @ARGV' "$@"; }
fi
timed_out() { [ "$1" = 124 ] || [ "$1" = 142 ]; }

# ------------------------------------------------------------------ defaults
OUT_MESH="out/meshes"
OUT_JSON="out/runs"
JOBS=4
TIMEOUT=300
BUDGET=250000
ITERS=8
FIDELITY=0
HEAL=0
VERBOSE=0
DRYRUN=0
FAILED_ONLY=0
RAW_LOG=0
BASE_FRAC=""
APPROVE=""

usage() {
  cat <<'EOF'
CADtoMesh_Agent — CAD to simulation-ready surface mesh

USAGE
  ./run.sh <command> [options] [cad files or directory]

COMMANDS
  setup                 create .venv and install dependencies (uses uv if present)
  sync                  make .venv match requirements.txt EXACTLY (uv only)
  test                  run the full test suite (300 tests, no gmsh required)
  parts                 generate the six synthetic test parts into models/
  doctor                check the environment: python, gmsh, packages, models

  stage0   [cad...]     geometry audit only — no meshing, fast triage
  backend  [cad...]     single mesh at default sizing, no adaptation
  mesh     [cad...]     FULL adaptive pipeline, writes .msh + .stl + manifest
  corpus   <dir>        many files: resumable, parallel, per-file timeout
  report   [dir]        aggregate ALL run records in a directory into corpus
                        statistics — outcomes, failures, timing. Runs nothing.

  compare               three research meshers vs gmsh on a shared boundary
  verify   [cad...]     gmsh API assumptions + CAD entity-ID stability
  calibrate [dir]       collect threshold findings and propose cut points
  view     <name>       open a produced mesh in the gmsh GUI
  log      [name]       what a run actually did: iterations, actions, rationale
                        no name lists every run; --failed narrows it

  all      [cad...]     parts -> stage0 -> mesh -> report
  clean                 delete generated output (never touches models/)

OPTIONS
  -o, --out DIR         mesh output directory      (default: out/meshes)
  -J, --json DIR        run-record directory       (default: out/runs)
  -j, --jobs N          parallel workers, corpus   (default: 4)
  -t, --timeout SEC     per-file timeout           (default: 300)
  -b, --budget N        element budget             (default: 250000)
  -i, --iterations N    max adaptive iterations    (default: 8)
      --base-frac F     initial element size as a fraction of model scale
      --fidelity        measure chordal/normal deviation (slow: OCC per sample)
      --heal            start with OCC healing enabled
      --failed          (log) list only runs that did not accept
      --raw             (log) print the backend's stdout instead of the record
      --approve PID     approve a healed-away CAD face (repeatable)
  -n, --dry-run         print the commands instead of running them
  -v, --verbose         pass through backend chatter
  -h, --help            this message

EXAMPLES
  ./run.sh all                                  everything, synthetic parts
  ./run.sh mesh models/block_hole.step --fidelity
  ./run.sh corpus models/abc -j 6 -t 180
  ./run.sh stage0 models/abc                    triage a directory, no meshing
  ./run.sh view block_hole                      open the result in gmsh
  ./run.sh log                                  index every run
  ./run.sh log --failed                         only the ones that did not accept
  ./run.sh log block_hole                       why that part ended where it did

If no CAD files are given, commands default to models/*.step and friends.

uv is used automatically when installed — it creates an ordinary .venv, which
this script activates for you on every invocation. PYTHON=3.12 ./run.sh setup
pins the interpreter version.
EOF
}

# ------------------------------------------------------------- option parsing
CMD="${1:-help}"; shift || true
FILES=()

while [ $# -gt 0 ]; do
  case "$1" in
    -o|--out)        OUT_MESH="$2"; shift 2 ;;
    -J|--json)       OUT_JSON="$2"; shift 2 ;;
    -j|--jobs)       JOBS="$2"; shift 2 ;;
    -t|--timeout)    TIMEOUT="$2"; shift 2 ;;
    -b|--budget)     BUDGET="$2"; shift 2 ;;
    -i|--iterations) ITERS="$2"; shift 2 ;;
    --base-frac)     BASE_FRAC="$2"; shift 2 ;;
    --approve)       APPROVE="$APPROVE --approve-removal $2"; shift 2 ;;
    --fidelity)      FIDELITY=1; shift ;;
    --failed)        FAILED_ONLY=1; shift ;;
    --raw)           RAW_LOG=1; shift ;;
    --heal)          HEAL=1; shift ;;
    -n|--dry-run)    DRYRUN=1; shift ;;
    -v|--verbose)    VERBOSE=1; shift ;;
    -h|--help)       usage; exit 0 ;;
    -*)              die "unknown option '$1' — try ./run.sh help" ;;
    *)               FILES+=("$1"); shift ;;
  esac
done

# Count files matching a glob. `ls glob | wc -l` is wrong twice over: with no
# match ls receives zero arguments and lists the current directory, and the
# unmatched pattern would count as one file. With nullglob global, the caller's
# glob is already gone, so the argument count is the answer.
count_glob() { printf '%s' "$#"; }

run() {                       # echo under --dry-run, execute otherwise
  if [ "$DRYRUN" = 1 ]; then printf '%s  %s%s\n' "$DIM" "$*" "$R"; return 0; fi
  "$@"
}

# Expand the positional arguments into a list of CAD files. A directory is
# scanned; nothing at all falls back to models/. Case-insensitive extensions,
# because ABC ships .STEP alongside .step.
collect_cad() {
  CAD=()
  local roots=()
  if [ ${#FILES[@]} -eq 0 ]; then roots=(models); else roots=("${FILES[@]}"); fi
  local r f
  for r in "${roots[@]}"; do
    if [ -d "$r" ]; then
      for f in "$r"/*.step "$r"/*.stp "$r"/*.STEP "$r"/*.STP \
               "$r"/*.iges "$r"/*.igs "$r"/*.brep; do
        CAD+=("$f")
      done
    elif [ -f "$r" ]; then
      CAD+=("$r")
    else
      warn "not found: $r"
    fi
  done
  [ ${#CAD[@]} -gt 0 ] || die "no CAD files found. Try: ./run.sh parts"
}

pipeline_flags() {
  FLAGS="--out $OUT_MESH --budget $BUDGET --max-iterations $ITERS"
  [ "$FIDELITY" = 1 ] || FLAGS="$FLAGS --no-fidelity"
  [ -n "$BASE_FRAC" ] && FLAGS="$FLAGS --base-frac $BASE_FRAC"
  [ -n "$APPROVE" ]   && FLAGS="$FLAGS$APPROVE"
  [ "$VERBOSE" = 1 ]  && FLAGS="$FLAGS -v"
  return 0
}

need_gmsh() {
  [ "$DRYRUN" = 1 ] && return 0     # a dry run prints commands; it needs nothing
  "${PY_CMD[@]}" -c "import gmsh" 2>/dev/null && return 0
  die "gmsh is not installed in this environment. Run: ./run.sh setup"
}

# Run one python module per file, with a timeout, and tally the outcomes.
# Per file rather than one invocation over a glob: a single pathological part
# must not be able to consume the whole run.
each_file() {                 # each_file <label> <module> <extra flags...>
  local label="$1"; shift
  local module="$1"; shift
  local n=${#CAD[@]} i=0 rc pass=0 fail=0 slow=0
  head_ "$label — $n file(s), ${TIMEOUT}s each"
  for f in "${CAD[@]}"; do
    i=$((i + 1))
    local stem; stem="$(basename "$f")"; stem="${stem%.*}"
    printf '%s[%2d/%2d]%s %-46s ' "$DIM" "$i" "$n" "$R" "${stem:0:46}"
    if [ "$DRYRUN" = 1 ]; then
      printf '\n%s    %s -m %s %s %s --json %s%s\n' \
        "$DIM" "$PY_SHOW" "$module" "$f" "$*" "$OUT_JSON/$stem.json" "$R"
      continue
    fi
    _TO "$TIMEOUT" "${PY_CMD[@]}" -m "$module" "$f" "$@" \
        --json "$OUT_JSON/$stem.json" >"$OUT_JSON/$stem.log" 2>&1
    rc=$?
    if timed_out $rc;     then printf '%stimeout%s\n' "$YEL" "$R"; slow=$((slow+1))
    elif [ $rc -eq 0 ];   then printf '%sok%s\n'      "$GRN" "$R"; pass=$((pass+1))
    else                       printf '%sattention (rc=%s)%s\n' "$YEL" "$rc" "$R"
                               fail=$((fail+1)); fi
  done
  [ "$DRYRUN" = 1 ] && return 0
  say ""
  ok "$pass ok, $fail need attention, $slow timed out"
  say "${DIM}logs and records in $OUT_JSON/${R}"
}

# -------------------------------------------------------------------- commands
case "$CMD" in

  help|-h|--help) usage ;;

  setup)
    head_ "Setting up"
    if [ -n "$UV" ]; then
      say "${DIM}using uv ($("$UV" --version 2>/dev/null))${R}"
      # uv venv is idempotent and produces a standard venv, so the activation
      # at the top of this script works unchanged. PYTHON=3.12 pins the version.
      if [ -n "${PYTHON:-}" ]; then run uv venv --python "$PYTHON"
      else                          run uv venv; fi
      [ "$DRYRUN" = 1 ] || . .venv/bin/activate
      run uv pip install -r requirements.txt \
        || die "dependency install failed — see the error above. \
Common causes: no network, or a package with no wheel for this platform."
    else
      warn "uv not found — falling back to venv + pip (slower)"
      say "${DIM}install uv: curl -LsSf https://astral.sh/uv/install.sh | sh${R}"
      [ -d .venv ] || run "${PY_CMD[@]}" -m venv .venv
      # shellcheck disable=SC1091
      [ "$DRYRUN" = 1 ] || . .venv/bin/activate
      run pip install -q --upgrade pip
      run pip install -q -r requirements.txt \
        || die "dependency install failed — see the error above."
    fi
    if [ "$DRYRUN" != 1 ]; then
      "${PY_CMD[@]}" -c "import numpy" 2>/dev/null \
        || die "install reported success but numpy is not importable — \
check that .venv is the environment being used (./run.sh doctor)."
    fi
    ok "environment ready — activate with: source .venv/bin/activate"
    say "${DIM}or just use ./run.sh, which activates .venv for you${R}"
    ;;

  sync)
    # Stricter than setup: uv pip sync makes the venv EXACTLY requirements.txt,
    # removing anything not listed. Useful when a stale package is suspected.
    [ -n "$UV" ] || die "sync needs uv — install it, or use ./run.sh setup"
    head_ "Syncing .venv to requirements.txt exactly"
    [ -d .venv ] || run uv venv
    run uv pip sync requirements.txt || die "sync failed — see the error above."
    ok "venv now matches requirements.txt exactly"
    ;;

  doctor)
    head_ "Environment"
    printf '  python      %s\n' "$("${PY_CMD[@]}" --version 2>&1)"
    printf '  runner      %s\n' "$PY_SHOW"
    if [ -n "$UV" ]; then printf '  uv          %s\n' "$("$UV" --version 2>&1)"
    else printf '  uv          %snot installed%s (setup falls back to pip)\n' "$YEL" "$R"; fi
    printf '  venv        %s\n' "${VIRTUAL_ENV:-not active}"
    if [ -f .venv/pyvenv.cfg ] && grep -qi '^uv *=' .venv/pyvenv.cfg 2>/dev/null; then
      printf '  venv built  %sby uv%s\n' "$GRN" "$R"
    elif [ -f .venv/pyvenv.cfg ]; then
      printf '  venv built  by python -m venv\n'
    fi
    for m in gmsh numpy trimesh scipy rtree; do
      v=$("${PY_CMD[@]}" -c "import $m,sys;print(getattr($m,'__version__','?'))" 2>/dev/null) \
        && printf '  %-11s %s\n' "$m" "$v" \
        || printf '  %-11s %sMISSING%s\n' "$m" "$RED" "$R"
    done
    printf '  models/     %s CAD file(s)\n' \
      "$(count_glob models/*.step models/*.stp models/*.STEP models/*.STP)"
    printf '  meshes      %s .msh produced\n' "$(count_glob "$OUT_MESH"/*.msh)"
    ;;

  test)
    head_ "Test suite"
    total=0; failed=0
    for t in tests/test_*.py; do
      line=$("${PY_CMD[@]}" "$t" 2>&1 | tail -1)
      n=$(printf '%s' "$line" | awk '{print $1}')
      case "$line" in *"0 failed"*) mark="$GRN✓$R" ;; *) mark="$RED✗$R"; failed=$((failed+1)) ;; esac
      printf '  %b %-34s %s\n' "$mark" "$(basename "$t")" "$line"
      case "$n" in ''|*[!0-9]*) ;; *) total=$((total + n)) ;; esac
    done
    say ""
    [ "$failed" -eq 0 ] && ok "$total tests passed" || die "$failed file(s) had failures"
    ;;

  parts)
    need_gmsh
    head_ "Generating synthetic parts"
    run "${PY_CMD[@]}" -m tools.make_test_parts models/
    ;;

  stage0)
    need_gmsh; collect_cad; mkdir -p "$OUT_JSON"
    each_file "Stage 0 geometry audit" stage0.run
    ;;

  backend)
    need_gmsh; collect_cad; mkdir -p "$OUT_JSON"
    extra=""; [ "$FIDELITY" = 1 ] && extra="--fidelity"
    each_file "Backend single-shot mesh" tools.verify_backend $extra
    ;;

  mesh)
    need_gmsh; collect_cad; pipeline_flags
    mkdir -p "$OUT_MESH" "$OUT_JSON"
    each_file "Adaptive pipeline" tools.run_pipeline $FLAGS -q
    say "${DIM}meshes: $(count_glob "$OUT_MESH"/*.msh) .msh in $OUT_MESH/${R}"
    ;;

  corpus)
    need_gmsh
    [ ${#FILES[@]} -gt 0 ] || die "corpus needs a directory: ./run.sh corpus models/abc"
    dir="${FILES[0]}"
    [ -d "$dir" ] || die "not a directory: $dir"
    head_ "Corpus run — $dir"
    extra=""; [ "$FIDELITY" = 1 ] && extra="--fidelity"
    run "${PY_CMD[@]}" -m tools.corpus "$dir" --out "$OUT_JSON" --meshes "$OUT_MESH" \
        -j "$JOBS" --timeout "$TIMEOUT" --budget "$BUDGET" \
        --max-iterations "$ITERS" $extra
    ;;

  report)
    dir="${FILES[0]:-$OUT_JSON}"
    [ -d "$dir" ] || die "no records at $dir — run ./run.sh mesh or corpus first"
    run "${PY_CMD[@]}" -m tools.corpus --out "$dir" --meshes "$OUT_MESH" --report
    ;;

  compare)
    need_gmsh
    head_ "Research meshers vs gmsh"
    run "${PY_CMD[@]}" -m tools.compare_meshers --gmsh --out "$OUT_MESH" --svg
    say ""
    say "${DIM}the metric-normalization ablation:${R}"
    run "${PY_CMD[@]}" -m tools.compare_meshers --case cylinder --target 1.0 --no-linearize
    ;;

  verify)
    need_gmsh; collect_cad
    head_ "gmsh API assumptions"
    run "${PY_CMD[@]}" -m tools.verify_gmsh_api models/cylinder.step --radius 12.0
    head_ "CAD entity-ID stability"
    run "${PY_CMD[@]}" -m tools.verify_entity_ids "${CAD[@]}"
    ;;

  calibrate)
    dir="${FILES[0]:-$OUT_JSON}"
    head_ "Threshold calibration"
    run "${PY_CMD[@]}" -m tools.calibrate collect "$dir" --csv calibration.csv
    say ""
    say "${CYN}Fill the 'verdict' column in calibration.csv with real / false,${R}"
    say "${CYN}then: ./run.sh calibrate --  (re-runs fit)${R}"
    [ -f calibration.csv ] && run "${PY_CMD[@]}" -m tools.calibrate fit calibration.csv
    ;;

  log|logs)
    # The run RECORD, not the raw stdout — it carries the per-iteration evidence
    # and the action taken with its rationale, which is what explains a run.
    extra=""
    [ "$FAILED_ONLY" = 1 ] && extra="--failed"
    [ "$RAW_LOG" = 1 ] && extra="$extra --raw"
    if [ ${#FILES[@]} -gt 0 ]; then
      run "${PY_CMD[@]}" -m tools.show_run "${FILES[0]}" --dir "$OUT_JSON" $extra
    else
      run "${PY_CMD[@]}" -m tools.show_run --dir "$OUT_JSON" $extra
    fi
    ;;

  view)
    [ ${#FILES[@]} -gt 0 ] || die "which mesh? e.g. ./run.sh view block_hole"
    name="${FILES[0]}"; name="${name%.msh}"
    msh="$OUT_MESH/$name.msh"
    [ -f "$msh" ] || die "no mesh at $msh — run ./run.sh mesh first"
    say "${DIM}physical groups = CAD faces. Tools > Statistics for quality,${R}"
    say "${DIM}Tools > Visibility to isolate one face.${R}"
    if command -v gmsh >/dev/null 2>&1; then run gmsh "$msh"
    else run "${PY_CMD[@]}" -c "import gmsh;gmsh.initialize();gmsh.open('$msh');gmsh.fltk.run();gmsh.finalize()"
    fi
    ;;

  all)
    need_gmsh
    # Flags this stage was invoked with must reach the sub-invocations, or
    # `all --dry-run` silently runs for real and `all --fidelity` silently does
    # not measure fidelity.
    PASS=""
    [ "$DRYRUN" = 1 ]   && PASS="$PASS -n"
    [ "$VERBOSE" = 1 ]  && PASS="$PASS -v"
    [ "$FIDELITY" = 1 ] && PASS="$PASS --fidelity"
    [ "$HEAL" = 1 ]     && PASS="$PASS --heal"
    [ -n "$BASE_FRAC" ] && PASS="$PASS --base-frac $BASE_FRAC"

    n=$(count_glob models/*.step models/*.stp)
    [ "$n" -gt 0 ] || "$0" parts $PASS
    "$0" stage0 "${FILES[@]}" -J "$OUT_JSON" -t "$TIMEOUT" $PASS || true
    "$0" mesh   "${FILES[@]}" -o "$OUT_MESH" -J "$OUT_JSON" \
                -b "$BUDGET" -i "$ITERS" -t "$TIMEOUT" $PASS || true
    "$0" report -J "$OUT_JSON" -o "$OUT_MESH" $PASS
    ;;

  clean)
    head_ "Cleaning generated output"
    run rm -rf out/ calibration.csv
    ok "removed out/ and calibration.csv — models/ untouched"
    ;;

  *) die "unknown command '$CMD' — try ./run.sh help" ;;
esac
