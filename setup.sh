#!/bin/bash
# Setup script for AWS EC2 Ubuntu - Breast Cancer Detection Training
# Run this after SSH into your EC2 instance

set -e  # Exit on error

echo "========================================="
echo "Breast Cancer Detection - EC2 Setup"
echo "========================================="

# Update system
echo "[1/8] Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install essential packages
echo "[2/8] Installing Python, pip, and git..."
sudo apt install -y python3-pip python3-venv git wget curl tmux htop nvtop

# Install NVIDIA drivers and CUDA
echo "[3/8] Installing NVIDIA drivers and CUDA..."
sudo apt install -y nvidia-driver-535 nvidia-cuda-toolkit

# Verify GPU
echo "[4/8] Verifying GPU..."
nvidia-smi

# Create workspace
echo "[5/8] Creating workspace..."
mkdir -p ~/workspace
cd ~/workspace

# Clone repository (user will do this manually with their credentials)
echo "[6/8] Ready for git clone"
echo "    Run: git clone https://github.com/YOUR_USERNAME/breast-cancer-detection.git"
echo "    Then: cd breast-cancer-detection"

# Install Python dependencies
echo "[7/8] Installing Python packages..."
python3 -m pip install --upgrade pip

# Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Detectron2
pip install git+https://github.com/facebookresearch/detectron2.git

# Install other requirements
echo "[8/8] Installing project requirements..."
# requirements.txt will be installed after git clone

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next Steps:"
echo "1. Clone your repository"
echo "2. cd breast-cancer-detection"
echo "3. pip install -r requirements.txt"
echo "4. Download dataset: ./download_data.sh"
echo "5. Convert dataset: python convert_dataset.py"
echo "6. Start training: ./train.sh"
echo ""
