#!/bin/bash

# Script to download checkpoints from the official Google Drive links released in https://github.com/yang-song/score_sde_pytorch#pretrained-models

# Target directory
CIFAR10_PRETRAIN_TARGET_DIR="checkpoints/subvp/cifar10_ddpmpp_deep_continuous"

# Create target directory if it doesn't exist
mkdir -p "$CIFAR10_PRETRAIN_TARGET_DIR"

# Download the file into the target directory
gdown --output "$CIFAR10_PRETRAIN_TARGET_DIR/model.pth" "https://drive.google.com/uc?id=1r8rgOfvgMWP2S48S2_AbmW--LvyjNfim"
