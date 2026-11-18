# Sourced by every job. EDIT THIS FILE for your JHPCE environment.
# Jobs must be submitted from the repo root (sbatch sets SLURM_SUBMIT_DIR to it).
# The .sbatch files use `set -euo pipefail`, but ~/.bash_profile and conda's activate
# scripts read unset variables (e.g. SYS_SYSROOT) and would abort. Relax -e/-u while
# loading them, then restore the caller's options below.
case $- in *e*) _had_e=1;; esac
case $- in *u*) _had_u=1;; esac
set +eu
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# --- Python environment (edit) -------------------------------------------
# Needs: torch (CUDA build), numpy, scipy, pandas, scikit-learn, matplotlib
# (requirements.txt; the paper used torch 2.5.1+cu121).
# If conda is not on PATH in batch jobs, add the appropriate `module load ...` line here.
CONDA_ENV="${CONDA_ENV:-tss}"
# Preferred: use the env's python directly. `conda shell.bash hook` / `conda activate`
# HUNG for 19+ min in a batch job on JHPCE, and a pip-installed torch env needs no activation.
# Override the location with CONDA_ENV_PREFIX=/path/to/env
_envdir="${CONDA_ENV_PREFIX:-$HOME/.conda/envs/$CONDA_ENV}"
if [ -x "$_envdir/bin/python" ]; then
  echo "using env python directly: $_envdir (skipping conda activate)"
  export PATH="$_envdir/bin:$PATH"
  export CONDA_DEFAULT_ENV="$CONDA_ENV"
else
  echo "env not found at $_envdir; falling back to conda activate"
  # 1) conda not visible in the batch job? -> load the login profile (module load conda_R ...)
  if ! command -v conda >/dev/null 2>&1 && [ -f ~/.bash_profile ]; then
    echo "conda not on PATH; sourcing ~/.bash_profile"
    source ~/.bash_profile
  fi
  # 2) still not? -> source conda's own init script (override with CONDA_SH=/path/to/conda.sh)
  if ! command -v conda >/dev/null 2>&1; then
    for _f in "${CONDA_SH:-}" /jhpce/shared/community/core/conda_R/4.3.x/etc/profile.d/conda.sh; do
      if [ -n "$_f" ] && [ -f "$_f" ]; then echo "sourcing $_f"; source "$_f"; break; fi
    done
    unset _f
  fi
  echo "conda: $(command -v conda || echo 'NOT FOUND')"
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV" || echo "WARNING: 'conda activate $CONDA_ENV' failed (is the env name right? CONDA_ENV=$CONDA_ENV)"
  fi
  # 3) last resort: use the env's python directly (no activation scripts)
  if [ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ] && [ -x "$HOME/.conda/envs/$CONDA_ENV/bin/python" ]; then
    echo "WARNING: falling back to PATH-only activation of $HOME/.conda/envs/$CONDA_ENV"
    export PATH="$HOME/.conda/envs/$CONDA_ENV/bin:$PATH"
    export CONDA_DEFAULT_ENV="$CONDA_ENV"
  fi
fi
unset _envdir
[ -n "${_had_e:-}" ] && set -e   # restore the caller's set -e / set -u
[ -n "${_had_u:-}" ] && set -u
unset _had_e _had_u
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
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    import sys
    maj, mn = torch.cuda.get_device_capability()
    archs = [a for a in torch.cuda.get_arch_list() if a.startswith("sm_")]
    # A cubin built for sm_<maj><k> runs on a device with the same major and k <= its minor.
    ok = any(int(a[3:-1]) == maj and int(a[-1]) <= mn for a in archs)
    print("GPU:", torch.cuda.get_device_name(0), f"(CC {maj}.{mn}) | torch built for:", archs)
    if not ok:
        sys.exit(f"ERROR: this torch build has no kernels for CC {maj}.{mn}. "
                 "Request a supported GPU: --gres=gpu:l40s:1 (not titanv/tesv100/tesv100s, CC 7.0).")
    print("cudnn:", torch.backends.cudnn.version())
PY
