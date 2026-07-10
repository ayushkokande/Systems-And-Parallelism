#!/bin/bash

SIZES=("small" "medium" "large" "xl" "2.7B")
CONTEXTS=(128 256 512 1024)

for SIZE in "${SIZES[@]}"; do
  for CTX in "${CONTEXTS[@]}"; do
    
    JOB_NAME="bench_${SIZE}_ctx${CTX}"
    FILE_NAME="submit_${JOB_NAME}.slurm"

    cat <<EOT > $FILE_NAME
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --account=YOUR_HPC_ACCOUNT
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=./logs/${JOB_NAME}_%j.out
#SBATCH --error=./logs/${JOB_NAME}_%j.err

singularity exec --bind /scratch --nv \\
--overlay /scratch/$USER/overlay-25GB-500K.ext3:ro \\
/scratch/$USER/ubuntu-20.04.3.sif \\
/bin/bash -c "

source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:\$PATH
export PATH=/scratch/$USER/tools/bin:\$PATH
export UV_CACHE_DIR=/scratch/$USER/.uv_cache

set -euo pipefail
cd /scratch/$USER/Systems-And-Parallelism

echo \"=== Benchmarking Size: $SIZE | Context: $CTX ===\"

uv run python -m systems.benchmark \\
  --size $SIZE \\
  --context-length $CTX \\
  --warmup-steps 5 \\
  --measure-steps 10 \\
  --mixed-precision \\
  --output-csv benchmark_${SIZE}_ctx${CTX}_bf16.csv
"
EOT

    echo "Created $FILE_NAME"
    # Optional: Uncomment the next line to submit automatically
    sbatch $FILE_NAME
  done
done