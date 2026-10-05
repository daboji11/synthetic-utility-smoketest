#!/usr/bin/env bash
# WSL/Linux only. Requires a working RoboCasa CUDA base environment and simulator assets.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_EXE:-$HOME/miniconda3/bin/conda}"
BASE_ENV="${BASE_ENV:-synutility-sim}"
TARGET_ENV="${TARGET_ENV:-synutility-dp-pretrained}"
SOURCE_ROOT="${DP_SOURCE_ROOT:-$HOME/research}"
mkdir -p "$SOURCE_ROOT"
clone_pinned() {
    local url="$1" destination="$2" commit="$3"
    if [ ! -d "$destination" ]; then
        git clone "$url" "$destination"
        git -C "$destination" checkout --detach "$commit"
    fi
    if [ "$(git -C "$destination" rev-parse HEAD)" != "$commit" ]; then
        echo "Existing source has a different commit: $destination. Preserving it." >&2
        return 1
    fi
}
clone_pinned https://github.com/robocasa-benchmark/diffusion_policy.git "$SOURCE_ROOT/diffusion_policy" 41212698b6f481ed92a55f0d5f1778ec56bea417
clone_pinned https://github.com/robocasa-benchmark/robomimic.git "$SOURCE_ROOT/robomimic-dp" 7dc18461c4fc497095f1044f4c40cd44fff62e6a
if ! "$CONDA_BIN" run -n "$TARGET_ENV" python --version >/dev/null 2>&1; then
    "$CONDA_BIN" create -n "$TARGET_ENV" --clone "$BASE_ENV" -y
fi
"$CONDA_BIN" run -n "$TARGET_ENV" python -m pip install --no-deps -e "$SOURCE_ROOT/diffusion_policy" -e "$SOURCE_ROOT/robomimic-dp"
"$CONDA_BIN" run -n "$TARGET_ENV" python -m pip install --no-deps lerobot==0.3.3
"$CONDA_BIN" run -n "$TARGET_ENV" python -m pip install -r "$ROOT/requirements-native-dp.txt"
# The upstream setup.py uses find_packages, while diffusion_policy is a namespace package.
"$CONDA_BIN" run -n "$TARGET_ENV" python -c 'import sysconfig,sys; from pathlib import Path; Path(sysconfig.get_paths()["purelib"],"robocasa_official_dp.pth").write_text(sys.argv[1]+"\n")' "$SOURCE_ROOT/diffusion_policy"
MUJOCO_GL=egl "$CONDA_BIN" run -n "$TARGET_ENV" python -c 'import torch; from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import DiffusionTransformerHybridImagePolicy; assert torch.cuda.is_available(); print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0))'
