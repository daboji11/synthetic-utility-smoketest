#!/usr/bin/env bash
# Linux/WSL only. Creates a standalone native DP env if no existing base env is present.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
if [ -z "$CONDA_BIN" ] || [ ! -x "$CONDA_BIN" ]; then
    echo "Conda executable missing; set CONDA_EXE to your Linux conda executable." >&2
    exit 1
fi
BASE_ENV="${BASE_ENV:-synutility-sim}"
TARGET_ENV="${TARGET_ENV:-synutility-dp-pretrained}"
SOURCE_ROOT="${DP_SOURCE_ROOT:-$HOME/research}"
export MUJOCO_GL=egl
export PYNPUT_BACKEND=dummy
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
if ! "$CONDA_BIN" run -n "$TARGET_ENV" python --version >/dev/null 2>&1; then
    if "$CONDA_BIN" run -n "$BASE_ENV" python --version >/dev/null 2>&1; then
        "$CONDA_BIN" create -n "$TARGET_ENV" --clone "$BASE_ENV" -y
    else
        "$CONDA_BIN" create -n "$TARGET_ENV" python=3.12 pip -y
    fi
fi
# Keep a working CUDA torch. New environments default to the official2.7.1 cu118 wheel.
if ! "$CONDA_BIN" run -n "$TARGET_ENV" python -c 'import torch,torchvision; assert torch.cuda.is_available()' >/dev/null 2>&1; then
    "$CONDA_BIN" run -n "$TARGET_ENV" python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url "${PYTORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu118}"
fi
"$CONDA_BIN" run -n "$TARGET_ENV" python -m pip install -r "$ROOT/requirements-native-dp.txt" -r "$ROOT/requirements-native-dp-base.txt"
clone_pinned https://github.com/ARISE-Initiative/robosuite.git "$SOURCE_ROOT/robosuite" 5ce6643f3092639d08f7b0f90ed1c6a84f50552c
clone_pinned https://github.com/robocasa/robocasa.git "$SOURCE_ROOT/robocasa" 456174f62b89b8fca99eaaf33949c29fec9cfc2a
clone_pinned https://github.com/robocasa-benchmark/diffusion_policy.git "$SOURCE_ROOT/diffusion_policy" 41212698b6f481ed92a55f0d5f1778ec56bea417
clone_pinned https://github.com/robocasa-benchmark/robomimic.git "$SOURCE_ROOT/robomimic-dp" 7dc18461c4fc497095f1044f4c40cd44fff62e6a
"$CONDA_BIN" run -n "$TARGET_ENV" python -m pip install --no-deps -e "$SOURCE_ROOT/robosuite" -e "$SOURCE_ROOT/robocasa" -e "$SOURCE_ROOT/diffusion_policy" -e "$SOURCE_ROOT/robomimic-dp" lerobot==0.3.3
# Namespace-package import fix; upstream setup.py omits namespace packages.
"$CONDA_BIN" run -n "$TARGET_ENV" python -c 'import sysconfig,sys; from pathlib import Path; Path(sysconfig.get_paths()["purelib"],"robocasa_official_dp.pth").write_text(sys.argv[1]+"\n")' "$SOURCE_ROOT/diffusion_policy"
if ! command -v ffmpeg >/dev/null 2>&1; then
    "$CONDA_BIN" install -n "$TARGET_ENV" -c conda-forge ffmpeg -y
fi
"$CONDA_BIN" run -n "$TARGET_ENV" python -c 'import torch; from diffusion_policy.workspace.train_diffusion_transformer_hybrid_workspace import TrainDiffusionTransformerHybridWorkspace; assert torch.cuda.is_available(); x=torch.randn(64,64,device="cuda"); y=x@x; print("SETUP_OK",torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0),y.shape)'
echo "Activate $TARGET_ENV; transfer the prepared28-demo dataset and download the pretrained checkpoint."
echo "Before rollout, download RoboCasa assets: python -m robocasa.scripts.download_kitchen_assets --type tex tex_generative fixtures_lw objs_lw"
