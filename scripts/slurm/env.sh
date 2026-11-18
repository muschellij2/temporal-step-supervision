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
# The same stale ~/.local cudnn can also be injected through LD_LIBRARY_PATH (e.g. by
# ~/.bash_profile), which PYTHONNOUSERSITE does not affect. Drop any ~/.local entries.
echo "LD_LIBRARY_PATH entries containing .local (removed): $(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep '/\.local/' || true)"
export LD_LIBRARY_PATH="$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '/\.local/' | paste -sd: - || true)"
python - <<'PY'
import torch
print("torch", torch.__version__, "from", torch.__file__)
print("cuda available:", torch.cuda.is_available(), "| cudnn:", torch.backends.cudnn.version())
if torch.cuda.is_available():
    import sys
    maj, mn = torch.cuda.get_device_capability()
    archs = [a for a in torch.cuda.get_arch_list() if a.startswith("sm_")]
    # A cubin built for sm_<maj><k> runs on a device with the same major and k <= its minor.
    ok = any(int(a[3:-1]) == maj and int(a[-1]) <= mn for a in archs)
    print("GPU:", torch.cuda.get_device_name(0), f"(CC {maj}.{mn}) | torch built for:", archs)
    if not ok:
        sys.exit(f"ERROR: this torch build has no kernels for CC {maj}.{mn}. "
                 "Request a newer GPU (e.g. the L40S) or install a torch build that supports it.")
PY
