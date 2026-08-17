#!/bin/zsh
setopt null_glob
if command -v gtimeout >/dev/null 2>&1; then run_to() { gtimeout "$@"; }
elif command -v timeout  >/dev/null 2>&1; then run_to() { timeout "$@"; }
else run_to() { perl -e 'alarm shift; exec @ARGV' "$@"; }
fi
LIMIT=${LIMIT:-120}
mkdir -p out/abc
for f in models/abc/*.{step,stp,STEP,STP}; do
  n=$(basename "$f"); n=${n%.*}
  run_to $LIMIT python -m stage0.run "$f" --json "out/abc/$n.stage0.json" \
      >"out/abc/$n.txt" 2>&1
  rc=$?
  case $rc in
    0)       s="clean" ;;
    2)       s="BLOCK: $(grep -m1 -A1 '^BLOCK' "out/abc/$n.txt" | tail -1 | xargs)" ;;
    124|142) s="TIMEOUT after ${LIMIT}s" ;;
    *)       s="ERROR($rc): $(tail -1 "out/abc/$n.txt" | cut -c1-55)" ;;
  esac
  printf '%-30s %s\n' "$n" "$s"
done
