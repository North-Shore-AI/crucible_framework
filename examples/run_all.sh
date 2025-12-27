#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

run_example() {
  local file="$1"
  echo
  echo "==> $file"
  (cd "$ROOT" && mix run "examples/$file")
}

run_example "01_core_pipeline.exs"
run_example "02_bench_optional.exs"
run_example "03_trace_optional.exs"
