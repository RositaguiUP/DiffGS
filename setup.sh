# cd nerfstudio
# pip install -e .
# cd ../
# pip install -e pcd_generator/depth_estimators/extern/depth_anything
# conda install pytorch=2.0.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# pip install -r requirements.txt
# cd realmdreamer
# pip install -e .
# cd ../
# pip install ./realmdreamer/realmdreamer/gaussian_splatting/simple-knn
# pip install ./realmdreamer/realmdreamer/gaussian_splatting/occlude
# pip install ./realmdreamer/diff-gaussian-rasterization
# pip uninstall -y tinycudann
# wget https://anaconda.org/pytorch3d/pytorch3d/0.7.7/download/linux-64/pytorch3d-0.7.7-py39_cu118_pyt201.tar.bz2
# conda install ./pytorch3d-0.7.7-py39_cu118_pyt201.tar.bz2

# wget --content-disposition -P ./pcd_generator/depth_estimators/checkpoints "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth"; wget --content-disposition -P ./pcd_generator/depth_estimators/checkpoints "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints_metric_depth/depth_anything_metric_depth_indoor.pt"

#!/bin/bash

# 1. Install Nerfstudio
cd nerfstudio
pip install -e .
cd ../

# 2. Install Depth Anything
pip install -e pcd_generator/depth_estimators/extern/depth_anything

# 3. Re-align PyTorch to System CUDA 12.x
# We use 12.1 as it is the most stable target for RTX 6000 Ada
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 4. Install requirements and build tools
pip install -r requirements.txt
pip install ninja Cython setuptools==69.5.1

# 5. Install RealmDreamer core
cd realmdreamer
pip install -e .
cd ../

# 6. Install CUDA extensions with NO BUILD ISOLATION
# This ensures they use the PyTorch 12.1 you just installed
export CUDA_HOME=/usr/local/cuda  # Ensure this points to your system CUDA 12.2
pip install ./realmdreamer/realmdreamer/gaussian_splatting/simple-knn --no-build-isolation
pip install ./realmdreamer/realmdreamer/gaussian_splatting/occlude --no-build-isolation
pip install ./realmdreamer/diff-gaussian-rasterization --no-build-isolation

# 7. Install tinycudann from source (REQUIRED for RTX 6000 Ada on CUDA 12.x)
pip uninstall -y tinycudann
# pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# 8. Install PyTorch3D (Fixed for CUDA 12.x)
# The old .tar.bz2 was for CUDA 11.8. We must install a compatible version.
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
cd ~/tests/diff/realmDreamer/
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d

# 8.1 Install from the local folder using NO isolation
# We also set the CUDA_HOME and FORCE_CUDA to ensure it builds for your GPU
export FORCE_CUDA=1
export CUDA_HOME=/usr/local/cuda  # Double check if this is where your CUDA 12.x is
pip install -e . --no-build-isolation

# 9. Download Checkpoints
wget --content-disposition -P ./pcd_generator/depth_estimators/checkpoints "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth"
wget --content-disposition -P ./pcd_generator/depth_estimators/checkpoints "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints_metric_depth/depth_anything_metric_depth_indoor.pt"