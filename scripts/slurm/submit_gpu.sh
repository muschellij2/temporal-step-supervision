#!/bin/bash
# Submit any scripts/slurm/*.sbatch to whichever of L40S / A100 / H100 is free.
#   bash scripts/slurm/submit_gpu.sh scripts/slurm/fit_all3.sbatch
#   bash scripts/slurm/submit_gpu.sh scripts/slurm/cv_marea.sbatch            # array job, all tasks on the chosen type
#   GPU_TYPES="l40s teslaa100" bash scripts/slurm/submit_gpu.sh <file> [extra sbatch args]
#   DRY=1 bash ...                                                           # print the sbatch command only
# sbatch cannot OR several --gres types, so this picks the type with the most idle nodes right now
# (ties -> order of GPU_TYPES) and passes --gres=gpu:<type>:1, which overrides the file's #SBATCH line.
# Check your site's exact gres names first:  sinfo -h -p gpu -o "%G" | sort -u
# (pattern below matches l40s / a100 / h100 case-insensitively, e.g. gpu:l40s, gpu:teslaa100, gpu:teslah100).
# Not V100/TitanV: torch's CC 7.0 kernels are not in the env (scripts/slurm/env.sh stops the job).
if [ "${BASH_SOURCE[0]}" != "$0" ]; then echo "Run with bash, do not source." >&2; return 1; fi
set -euo pipefail
file="${1:?usage: submit_gpu.sh <file.sbatch> [sbatch args]}"; shift
PART="${PARTITION:-bstgpu,bstgpu2,bstgpu3,gpu}"   # same list as the #SBATCH --partition lines
prefs="${GPU_TYPES:-l40s a100 h100}"

all=$(sinfo -h -p "$PART" -o "%G" | tr ',' '\n' | grep -oiE 'gpu:[A-Za-z0-9_]+' | cut -d: -f2 | sort -u)
best=""; bestn=-1
for pat in $prefs; do
  for t in $(echo "$all" | grep -i "$pat" || true); do
    n=$(sinfo -h -p "$PART" -t idle,mix -o "%G %D" | grep -i "gpu:$t" | awk '{s+=$NF} END{print s+0}')
    if [ "$n" -gt "$bestn" ]; then best="$t"; bestn="$n"; fi
  done
done
if [ -z "$best" ]; then echo "no l40s/a100/h100 gres found in partition $PART; available: $(echo $all)" >&2; exit 1; fi
echo "using gres gpu:$best:1 ($bestn nodes idle/mixed; candidates: $(echo $all))"
cmd=(sbatch --gres="gpu:$best:1" "$@" "$file")
if [ -n "${DRY:-}" ]; then echo "${cmd[@]}"; else "${cmd[@]}"; fi
