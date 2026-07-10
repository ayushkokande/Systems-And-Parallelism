#!/bin/bash
#SBATCH --job-name=benchmark_more_ctx
#SBATCH --account=YOUR_HPC_ACCOUNT
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=./logs/benchmark_more_%j.out
#SBATCH --error=./logs/benchmark_more_%j.err
#SBATCH --requeue

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


ctxs=(256 512 1024)

for c in \${ctxs[@]}; do
  echo \"=== Benchmarking all sizes | context=\$c | mixed precision ===\"

  uv run python -m systems.benchmark \
    --size all \
    --context-length \$c \
    --warmup-steps 5 \
    --measure-steps 10 \
    --mixed-precision \
    --output-csv benchmark_results_ctx\${c}_bf16.csv || echo \"FAILED: ctx=\$c\"

done

echo 'All BF16 benchmarking runs completed successfully!'
"