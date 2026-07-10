#!/bin/bash
#SBATCH --job-name=bench_large_ctx128
#SBATCH --account=YOUR_HPC_ACCOUNT
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=./logs/bench_large_ctx128_%j.out
#SBATCH --error=./logs/bench_large_ctx128_%j.err

singularity exec --bind /scratch --nv \
--overlay /scratch/$USER/overlay-25GB-500K.ext3:ro \
/scratch/$USER/ubuntu-20.04.3.sif \
/bin/bash -c "

source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:$PATH
export PATH=/scratch/$USER/tools/bin:$PATH
export UV_CACHE_DIR=/scratch/$USER/.uv_cache

set -euo pipefail
cd /scratch/$USER/Systems-And-Parallelism

echo \"=== Benchmarking Size: large | Context: 128 ===\"

uv run python -m systems.benchmark \
  --size large \
  --context-length 128 \
  --warmup-steps 5 \
  --measure-steps 10 \
  --mixed-precision \
  --output-csv benchmark_large_ctx128_bf16.csv
"
