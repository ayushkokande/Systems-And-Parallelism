#!/bin/bash
#SBATCH --job-name=nsys_profile
#SBATCH --account=csci_ga_3033_131-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=./logs/nsys_profile_%j.out
#SBATCH --error=./logs/nsys_profile_%j.err
#SBATCH --requeue

singularity exec --bind /scratch --bind /opt/nvidia/hpc_sdk --nv \
--overlay /scratch/ak13124/overlay-25GB-500K.ext3:ro \
/scratch/ak13124/ubuntu-20.04.3.sif \
/bin/bash -c "

source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:\$PATH
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/latest/profiler/bin:\$PATH
set -euo pipefail


cd /scratch/ak13124/a2


sizes=(small medium large xl 2.7B)
ctxs=(128 256 512 1024)

for s in \${sizes[@]}; do
  for c in \${ctxs[@]}; do
    echo \"=== \$s ctx=\$c ===\"

    uv run nsys profile -o nsys_reports/\${s}_ctx\${c}_fwd --trace=cuda,nvtx \
      python -m student.nsys_profile --size \$s --context-length \$c \
      --warmup-steps 5 --measure-steps 1 --forward-only --nvtx || echo \"FAILED: \$s ctx=\$c fwd\"

    uv run nsys profile -o nsys_reports/\${s}_ctx\${c}_fwbw --trace=cuda,nvtx \
      python -m student.nsys_profile --size \$s --context-length \$c \
      --warmup-steps 5 --measure-steps 1 --nvtx || echo \"FAILED: \$s ctx=\$c fwbw\"

    uv run nsys profile -o nsys_reports/\${s}_ctx\${c}_train --trace=cuda,nvtx \
      python -m student.nsys_profile --size \$s --context-length \$c \
      --warmup-steps 5 --measure-steps 1 --optimizer-step --nvtx || echo \"FAILED: \$s ctx=\$c train\"

  done
done

echo 'All nsys profiling runs completed!'
"
