#!/usr/bin/env bash
set -e
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -n uniforge3d python=3.10 -y
conda activate uniforge3d

python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install setuptools==80.9.0 wheel ninja cmake==3.31.6 packaging \
    transformers==4.57.3 peft==0.17.1 "huggingface-hub>=0.36,<1" \
    "timm>=1.0,<2" "kornia>=0.8,<0.9" "numpy>=2.1,<2.3" \
    "Pillow>=10.4,<12" "safetensors>=0.7,<1" trimesh==4.6.3 \
    "opencv-python-headless>=4.10,<5" "easydict>=1.13,<2" \
    "tqdm>=4.66,<5" "scipy>=1.13,<2"
python -m pip install flash-attn==2.7.3 --no-build-isolation

git clone --recursive https://github.com/microsoft/TRELLIS.2.git .deps/TRELLIS.2
git -C .deps/TRELLIS.2 checkout 75fbf0183001ed9876c8dbb35de6b68552ee08bd
git -C .deps/TRELLIS.2 submodule update --init --recursive

git clone --recursive https://github.com/JeffreyXiang/CuMesh.git .deps/CuMesh
git -C .deps/CuMesh checkout 12289e1062f0603f2f0d0771b02e1395d247f26f
git -C .deps/CuMesh submodule update --init --recursive

git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git .deps/FlexGEMM
git -C .deps/FlexGEMM checkout 6dd94a859c26ee8246888502eada3dd8ad85532e
git -C .deps/FlexGEMM submodule update --init --recursive

git clone --branch v0.4.0 https://github.com/NVlabs/nvdiffrast.git .deps/nvdiffrast

python -m pip install ./.deps/nvdiffrast --no-build-isolation
python -m pip install ./.deps/CuMesh --no-build-isolation
python -m pip install ./.deps/FlexGEMM --no-build-isolation
python -m pip install ./.deps/TRELLIS.2/o-voxel --no-build-isolation
