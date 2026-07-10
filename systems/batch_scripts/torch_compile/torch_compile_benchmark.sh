#!/bin/bash
#SBATCH --job-name=compile_bench
#SBATCH --account=YOUR_HPC_ACCOUNT
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=./logs/compile_bench_%j.out
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
uv run python -m systems.torch_compile_benchmark --output-csv-a compile_attn.csv --output-csv-b compile_model.csv
"
