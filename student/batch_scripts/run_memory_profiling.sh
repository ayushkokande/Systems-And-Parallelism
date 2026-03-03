#!/bin/bash
#SBATCH --job-name=mem_prof_2.7B
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=./logs/mem_prof_2.7B_%j.out
#SBATCH --error=./logs/mem_prof_2.7B_%j.err

singularity exec --bind /scratch --nv \
--overlay /scratch/ak13124/overlay-25GB-500K.ext3:ro \
/scratch/ak13124/ubuntu-20.04.3.sif \
/bin/bash -c "
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:\$PATH
export PATH=/scratch/ak13124/tools/bin:\$PATH
export UV_CACHE_DIR=/scratch/ak13124/.uv_cache

set -euo pipefail
cd /scratch/ak13124/a2/nyu-llm-reasoners-a2

echo \"=== Memory profiling: 2.7B model (forward-only) ===\"
for ctx in 128 256 512; do
  echo \"--- Context length: \${ctx} (forward) ---\"
  uv run python -m student.memory_profiling --size 2.7B --context-length \${ctx} --profile-memory --forward-only --warmup-steps 5 --measure-steps 1
done

echo \"=== Memory profiling: 2.7B model (full training step) ===\"
for ctx in 128 256 512; do
  echo \"--- Context length: \${ctx} (train) ---\"
  uv run python -m student.memory_profiling --size 2.7B --context-length \${ctx} --profile-memory --warmup-steps 5 --measure-steps 1
done

echo \"=== Done ===\"
"
