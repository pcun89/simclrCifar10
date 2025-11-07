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

