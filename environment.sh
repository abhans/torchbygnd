#!/bin/bash

# Install NVCC
sudo apt install nvidia-cuda-toolkit -y 

# Detect CUDA version
CUDA_VERSION=$(nvcc --version | grep -oP "(?<=V)\d+\.\d+" | head -1)

if [ -z "$CUDA_VERSION" ]; then
    echo "No CUDA installation detected. Exiting."
    exit 1
fi

# Anaconda Installation
ANACONDA_INSTALLER="Anaconda3-2024.10-1-Linux-x86_64.sh"
if [ ! -f "$ANACONDA_INSTALLER" ]; then
    echo "Downloading Anaconda installer..."
    curl -O https://repo.anaconda.com/archive/$ANACONDA_INSTALLER
    if [ $? -ne 0 ]; then
        echo "Failed to download Anaconda installer. Exiting."
        exit 1
    fi
fi

echo "Installing Anaconda..."
bash $ANACONDA_INSTALLER -b -p $HOME/anaconda3
if [ $? -ne 0 ]; then
    echo "Failed to install Anaconda. Exiting."
    exit 1
fi

# Initialize Conda (activate without modifying the shell's config files)
source $HOME/anaconda3/etc/profile.d/conda.sh
conda config --set auto_activate_base False

rm $ANACONDA_INSTALLER

# Create Conda environment
echo "Creating Conda environment..."
conda env create -f environment.yml
if [ $? -ne 0 ]; then
    echo "Failed to create Conda environment. Exiting."
    exit 1
fi

echo "Conda environment created successfully."