#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-cpu-maxpool-bench}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

docker build -f "${ROOT_DIR}/max_pooling/docker_benchmark_files/Dockerfile.cpu_bench" -t "${IMAGE_NAME}" "${ROOT_DIR}"
docker run --rm "${IMAGE_NAME}" "$@"
