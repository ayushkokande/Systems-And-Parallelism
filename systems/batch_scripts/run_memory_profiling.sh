#!/bin/bash
#SBATCH --job-name=mem_prof_2.7B
#SBATCH --account=YOUR_HPC_ACCOUNT
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=./logs/mem_prof_2.7B_%j.out
#SBATCH --error=./logs/mem_prof_2.7B_%j.err

singularity exec --bind /scratch --nv \
--overlay /scratch/$USER/overlay-25GB-500K.ext3:ro \
/scratch/$USER/ubuntu-20.04.3.sif \
/bin/bash -c "
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:\$PATH
export PATH=/scratch/$USER/tools/bin:\$PATH
export UV_CACHE_DIR=/scratch/$USER/.uv_cache

set -euo pipefail
cd /scratch/$USER/Systems-And-Parallelism

echo \"=== Memory profiling: 2.7B model (forward-only) ===\"
for ctx in 128 256 512; do
  echo \"--- Context length: \${ctx} (forward) ---\"
  uv run python -m systems.memory_profiling --size 2.7B --context-length \${ctx} --profile-memory --forward-only --warmup-steps 5 --measure-steps 1
done

echo \"=== Memory profiling: 2.7B model (full training step) ===\"
for ctx in 128 256 512; do
  echo \"--- Context length: \${ctx} (train) ---\"
  uv run python -m systems.memory_profiling --size 2.7B --context-length \${ctx} --profile-memory --warmup-steps 5 --measure-steps 1
done

echo \"=== Done ===\"
"
