# Sourced by every job. EDIT THIS FILE for your JHPCE environment.
# Jobs must be submitted from the repo root (sbatch sets SLURM_SUBMIT_DIR to it).
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the repo root}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# --- Python environment (edit) -------------------------------------------
# Needs: torch (CUDA build), numpy, scipy, pandas, scikit-learn, matplotlib
# (requirements.txt; the paper used torch 2.5.1+cu121).
# If conda is not on PATH in batch jobs, add the appropriate `module load ...` line here.
CONDA_ENV="${CONDA_ENV:-tss}"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
fi
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
