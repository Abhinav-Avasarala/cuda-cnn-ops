#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-cpu-conv2d-bench}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

docker build -f "${ROOT_DIR}/conv_2d/docker_benchmark_files/Dockerfile.cpu_bench" -t "${IMAGE_NAME}" "${ROOT_DIR}"
docker run --rm "${IMAGE_NAME}" "$@"

