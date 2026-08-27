#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Initialize submodules from the repository root, independent of cwd.
git -C "$REPO_ROOT" submodule update --init --recursive
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda create -n know_your_camera python=3.10 -y
conda activate know_your_camera

python -m pip install --upgrade pip setuptools wheel

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install numpy scipy h5py einops pillow tqdm imageio imageio-ffmpeg PyOpenGL glfw wandb pyyaml hydra-core==1.3.2

pip install diffusers transformers

pip install mujoco

# only need for robosuite
pip install -e "$REPO_ROOT/third_party/robosuite"

# only need for maniskill
pip install -e "$REPO_ROOT/third_party/maniskill"


# download demos
pip install gdown
mkdir -p "$REPO_ROOT/temp"
gdown --folder --remaining-ok --id 1dmv-ueaP8F0ElqgVXsdmX-S9hvfQb7Yf -O "$REPO_ROOT/temp"
mkdir -p "$SCRIPT_DIR/policy_maniskill/demos" "$SCRIPT_DIR/policy_robosuite/demos"
mv -n "$REPO_ROOT/temp/demos_maniskill/"* "$SCRIPT_DIR/policy_maniskill/demos/"
mv -n "$REPO_ROOT/temp/demos_robosuite/"* "$SCRIPT_DIR/policy_robosuite/demos/"
rm -rf "$REPO_ROOT/temp"


echo "Environment 'know_your_camera' is ready."
