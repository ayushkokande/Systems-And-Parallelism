#!/bin/bash
#SBATCH --job-name=compile_bench
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=./logs/compile_bench_%j.out
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
uv run python -m student.torch_compile_benchmark --output-csv-a compile_attn.csv --output-csv-b compile_model.csv
"
