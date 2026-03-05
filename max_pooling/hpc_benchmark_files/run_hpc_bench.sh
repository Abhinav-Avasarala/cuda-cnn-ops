#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_BIN="${ROOT_DIR}/bin/bench_all_max_pooling"

echo "Building via Makefile..."
(cd "${ROOT_DIR}" && make clean && make bench)

echo "Running: ${OUT_BIN} $*"
exec "${OUT_BIN}" "$@"
