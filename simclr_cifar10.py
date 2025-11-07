"""
simclr_cifar10.py

Self-supervised SimCLR-style training on CIFAR-10, then linear evaluation.
Single-file example for educational / resume use.

Usage:
    python simclr_cifar10.py --epochs_pretrain 50 --epochs_linear 20 --batch_size 256 --device cuda

Author: Generated for Phillipp (examples follow requested style)
"""

import argparse
import os
from typing import Tuple, Any, Dict
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models
from tqdm import tqdm

# -------------------------
# Utilities & Augmentations
# -------------------------


class TwoCropTransform:
    """
    Take two random crops of one image as the SimCLR paper does (two correlated views).
    """

    def __init__(self, baseTransform):
        self.baseTransform = baseTransform

    def __call__(self, x):
        return self.baseTransform(x), self.baseTransform(x)


def getSimCLRTransforms(imageSize: int = 32) -> transforms.Compose:
    """
    Return the augmentation pipeline used for SimCLR-like training on small images.
    """
    # For CIFAR-sized images use cropping, color jitter, grayscale, gaussian blur, horizontal flip
    colorJitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    transform = transforms.Compose([
        transforms.RandomResizedCrop(imageSize, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([colorJitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=3),  # small kernel for CIFAR
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.247, 0.243, 0.261))
    ])
    return transform
