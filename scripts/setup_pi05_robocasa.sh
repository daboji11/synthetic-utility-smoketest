#!/usr/bin/env bash
# Isolated Linux CUDA environment; never modifies native DP/ACT environments.
set -euo pipefail
export GIT_LFS_SKIP_SMUDGE=1
export UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=10
[ "$(uname -s)" = Linux ] || { echo "Run this on the Linux CUDA host (Mac is the SSH client)." >&2; exit 1; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ROOT="${PI05_SOURCE_ROOT:-$HOME/research}"
OPENPI_ROOT="${OPENPI_ROOT:-$SOURCE_ROOT/openpi-robocasa}"
UV_TOOL_ENV="${UV_TOOL_ENV:-$ROOT/.venv-pi05-tools}"
mkdir -p "$SOURCE_ROOT"
clone_pinned() {
 local url="$1" destination="$2" commit="$3"
 if [ ! -d "$destination" ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone --no-recurse-submodules "$url" "$destination"
  git -C "$destination" checkout --detach "$commit"
 fi
 [ "$(git -C "$destination" rev-parse HEAD)" = "$commit" ] || {
  echo "Preserving source with different revision: $destination" >&2; exit 1;
 }
}
clone_pinned https://github.com/robocasa-benchmark/openpi.git "$OPENPI_ROOT" 5a6beda9ff99da30b4e1b59320f6a32971d7c397
clone_pinned https://github.com/ARISE-Initiative/robosuite.git "$SOURCE_ROOT/robosuite" 5ce6643f3092639d08f7b0f90ed1c6a84f50552c
clone_pinned https://github.com/robocasa/robocasa.git "$SOURCE_ROOT/robocasa" 456174f62b89b8fca99eaaf33949c29fec9cfc2a
if [ ! -x "$UV_TOOL_ENV/bin/uv" ]; then
 BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-}"
 if [ -z "$BOOTSTRAP_PYTHON" ]; then
  for candidate in python3 "$HOME/miniconda3/bin/python" /workspace/miniforge3/bin/python; do
   if "$candidate" -c 'import ensurepip' >/dev/null 2>&1; then BOOTSTRAP_PYTHON="$candidate"; break; fi
  done
 fi
 [ -n "$BOOTSTRAP_PYTHON" ] || { echo "Set BOOTSTRAP_PYTHON to a Linux Python with ensurepip." >&2; exit 1; }
 "$BOOTSTRAP_PYTHON" -m venv "$UV_TOOL_ENV"
 "$UV_TOOL_ENV/bin/python" -m pip install uv==0.8.22
fi
UV="$UV_TOOL_ENV/bin/uv"
"$UV" sync --project "$OPENPI_ROOT" --frozen --no-dev --inexact --python 3.11
PY="$OPENPI_ROOT/.venv/bin/python"
"$UV" pip install --python "$PY" -r "$ROOT/requirements-pi05-runtime.txt"
"$UV" pip install --python "$PY" --no-deps -e "$SOURCE_ROOT/robosuite" -e "$SOURCE_ROOT/robocasa"
# The fork's uv Git pin reports LeRobot0.1.0 while both openpi and RoboCasa require0.3.3.
# Install the actually requested API; unused Aloha extra conflicts with RoboCasa MuJoCo.
"$UV" pip install --python "$PY" --no-deps lerobot==0.3.3
"$UV" pip uninstall --python "$PY" gym-aloha
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYNPUT_BACKEND=dummy HF_HUB_ENABLE_HF_TRANSFER=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
"$PY" -c 'import jax,torch; from openpi.training import config; devices=jax.devices(); assert any(d.platform=="gpu" for d in devices),devices; x=jax.numpy.ones((64,64)); (x@x).block_until_ready(); print("PI05_ENV_OK",devices,torch.__version__)'
"$UV" pip check --python "$PY"
"$UV" pip freeze --python "$PY" > "$ROOT/requirements-pi05-linux-lock.txt"
echo "Python: $PY"
echo "Assets (on a fresh host): $PY -m robocasa.scripts.download_kitchen_assets --type tex tex_generative fixtures_lw objs_lw"
echo "No weights, training or rollouts were started by setup."
