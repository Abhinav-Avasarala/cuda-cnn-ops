#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_BIN="${ROOT_DIR}/bin/bench_cpu_conv2d"

if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "Note: macOS often ships Apple clang as g++ and may not support OpenMP."
  echo "Recommended: use Docker via ./docker_bench.sh"
fi

echo "Building via Makefile..."
(cd "${ROOT_DIR}" && make clean && make bench)

echo "Running: ${OUT_BIN} $*"
exec "${OUT_BIN}" "$@"

