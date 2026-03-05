#!/usr/bin/env bash
#SBATCH --job-name=maxpool-all-bench
#SBATCH --output=maxpool-all-bench-%j.out
#SBATCH --error=maxpool-all-bench-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

set -euo pipefail

# Update module names for your cluster if needed.
module purge
module load gcc || true
module load cuda || true

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Keep OpenMP and Slurm CPU counts aligned by default.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-N/A}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"

bash "${ROOT_DIR}/run_hpc_bench.sh" "${OMP_NUM_THREADS}" 20 5
