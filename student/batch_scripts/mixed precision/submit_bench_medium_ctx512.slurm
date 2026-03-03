#!/bin/bash
#SBATCH --job-name=bench_medium_ctx512
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=./logs/bench_medium_ctx512_%j.out
#SBATCH --error=./logs/bench_medium_ctx512_%j.err

singularity exec --bind /scratch --nv \
--overlay /scratch/ak13124/overlay-25GB-500K.ext3:ro \
/scratch/ak13124/ubuntu-20.04.3.sif \
/bin/bash -c "

source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:$PATH
export PATH=/scratch/ak13124/tools/bin:$PATH
export UV_CACHE_DIR=/scratch/ak13124/.uv_cache

set -euo pipefail
cd /scratch/ak13124/a2/nyu-llm-reasoners-a2

echo \"=== Benchmarking Size: medium | Context: 512 ===\"

uv run python -m student.benchmark \
  --size medium \
  --context-length 512 \
  --warmup-steps 5 \
  --measure-steps 10 \
  --mixed-precision \
  --output-csv benchmark_medium_ctx512_bf16.csv
"
