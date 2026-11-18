# Sourced by every job. EDIT THIS FILE for your JHPCE environment.
# Jobs must be submitted from the repo root (sbatch sets SLURM_SUBMIT_DIR to it).
source ~/.bash_profile
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
# Fail loudly instead of silently running with the wrong python.
if [ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]; then
  echo "ERROR: conda env '$CONDA_ENV' is not active (CONDA_DEFAULT_ENV='${CONDA_DEFAULT_ENV:-}')." >&2
  echo "Add the right 'module load ...' line to scripts/slurm/env.sh or set CONDA_ENV=<name>." >&2
  exit 1
fi
echo "conda env: $CONDA_DEFAULT_ENV | python: $(command -v python)"
# Ignore ~/.local (user-level pip packages). A stale nvidia-cudnn there shadowed the
# env's copy and crashed with "undefined symbol: cudnnGetLibConfig".
export PYTHONNOUSERSITE=1
python - <<'PY'
import torch
print("torch", torch.__version__, "from", torch.__file__)
print("cuda available:", torch.cuda.is_available(), "| cudnn:", torch.backends.cudnn.version())
PY
