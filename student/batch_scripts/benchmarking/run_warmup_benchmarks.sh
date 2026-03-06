#!/bin/bash
#SBATCH --job-name=warmup_bench
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=./logs/warmup_%j.out
#SBATCH --error=./logs/warmup_%j.err
#SBATCH --requeue

singularity exec --bind /scratch --nv \
--overlay /scratch/ak13124/overlay-25GB-500K.ext3:ro \
/scratch/ak13124/ubuntu-20.04.3.sif \
/bin/bash -c "


source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:\$PATH
set -euo pipefail

cd /scratch/ak13124/a2

echo 'Starting warmup 0...'
uv run python -m student.benchmark --size all --context-length 128 --warmup-steps 0 --measure-steps 10 --output-csv benchmark_results_c_warmup0.csv

echo 'Starting warmup 1...'
uv run python -m student.benchmark --size all --context-length 128 --warmup-steps 1 --measure-steps 10 --output-csv benchmark_results_c_warmup1.csv

echo 'Starting warmup 2...'
uv run python -m student.benchmark --size all --context-length 128 --warmup-steps 2 --measure-steps 10 --output-csv benchmark_results_c_warmup2.csv

echo 'All benchmarks completed successfully!'
"