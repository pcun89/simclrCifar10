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


# -------------------------
# Model: encoder + head
# -------------------------
class ResNetSimCLR(nn.Module):
    """
    ResNet-based encoder with a projection head for contrastive learning (SimCLR).
    We use torchvision's ResNet18 and replace the final FC with identity, then add a projection MLP.
    """

    def __init__(self, baseModel: str = "resnet18", projectionDim: int = 128):
        super().__init__()
        if baseModel != "resnet18":
            raise ValueError("This example currently supports resnet18 only.")
        backbone = models.resnet18(pretrained=False)
        # remove final classifier
        numFeat = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        # projection head: MLP
        self.projection = nn.Sequential(
            nn.Linear(numFeat, numFeat),
            nn.ReLU(inplace=True),
            nn.Linear(numFeat, projectionDim)
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Return (representation, projection). representation is the backbone output (pre-projection).
        """
        rep = self.backbone(x)
        proj = self.projection(rep)
        return rep, proj


# -------------------------
# Loss: NT-Xent (normalized temperature-scaled cross entropy)
# -------------------------
class NTXentLoss(nn.Module):
    """
    Compute NT-Xent loss for a batch of 2N projected features (2 views per sample).
    Implementation assumes input z of shape (2N, dim) where first N are view1, next N are view2
    or interleaved. We'll implement using similarity matrix and masking.
    """

    def __init__(self, batchSize: int, temperature: float = 0.5, device: str = "cpu"):
        super().__init__()
        self.batchSize = batchSize
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        # create mask to remove similarity of sample with itself
        N = 2 * batchSize
        mask = torch.ones((N, N), dtype=bool)
        for i in range(batchSize):
            mask[i, i] = False
            mask[batchSize + i, batchSize + i] = False
            # remove positive pairs on cross positions? We'll handle positives separately
        self.register_buffer("mask", mask)

    def forward(self, z: Tensor) -> Tensor:
        """
        z: Tensor with shape (2*B, D)
        returns scalar loss
        """
        # normalize
        z = F.normalize(z, p=2, dim=1)
        N = z.shape[0]
        # similarity matrix
        sim = torch.matmul(z, z.T) / self.temperature  # (2B,2B)
        # positive pairs: i with i+B and vice versa
        batchSize = N // 2
        positives = torch.cat(
            [torch.diag(sim, batchSize), torch.diag(sim, -batchSize)]).view(N, 1)
        # negatives: mask out self and positive
        negMask = self.mask.clone()
        # build logits: for each example, logits = [positive, negatives...]
        logits = torch.cat([positives, sim[~negMask].view(N, -1)], dim=1)
        labels = torch.zeros(N, dtype=torch.long, device=self.device)
        loss = self.criterion(logits, labels)
        loss = loss / N
        return loss


# -------------------------
# Training loops
# -------------------------
def pretrainSimCLR(model: nn.Module,
                   dataLoader: DataLoader,
                   optimizer: torch.optim.Optimizer,
                   criterion: NTXentLoss,
                   device: str,
                   epoch: int,
                   logInterval: int = 50) -> float:
    """
    One epoch pretraining for SimCLR-style contrastive learning.
    Returns average loss.
    """
    model.train()
    totalLoss = 0.0
    it = 0
    for batch in tqdm(dataLoader, desc=f"Pretrain Epoch {epoch}"):
        (x1, x2), _ = batch  # dataset returns two views and labels (labels ignored)
        x1 = x1.to(device)
        x2 = x2.to(device)
        optimizer.zero_grad()
        _, z1 = model(x1)
        _, z2 = model(x2)
        z = torch.cat([z1, z2], dim=0)
        loss = criterion(z)
        loss.backward()
        optimizer.step()
        totalLoss += loss.item()
        it += 1
        if it % logInterval == 0:
            tqdm.write(f"Iter {it}, Loss {loss.item():.4f}")
    avgLoss = totalLoss / (it if it > 0 else 1)
    return avgLoss


def trainLinearEvaluation(encoder: nn.Module,
                          classifier: nn.Module,
                          trainLoader: DataLoader,
                          valLoader: DataLoader,
                          optimizer: torch.optim.Optimizer,
                          device: str,
                          epochs: int = 10) -> Dict[str, Any]:
    """
    Train a linear classifier on frozen encoder representations.
    Returns best validation accuracy and training history.
    """
    encoder.eval()  # freeze encoder
    classifier.train()
    history = {"trainLoss": [], "valAcc": []}
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        totalLoss = 0.0
        it = 0
        for images, targets in tqdm(trainLoader, desc=f"Linear Train Epoch {epoch+1}/{epochs}"):
            images = images.to(device)
            targets = targets.to(device)
            with torch.no_grad():
                reps, _ = encoder(images)  # rep shape (B, featDim)
            logits = classifier(reps)
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totalLoss += loss.item()
            it += 1
        avgTrainLoss = totalLoss / (it if it > 0 else 1)
        history["trainLoss"].append(avgTrainLoss)
        # validation
        correct = 0
        total = 0
        classifier.eval()
        with torch.no_grad():
            for images, targets in valLoader:
                images = images.to(device)
                targets = targets.to(device)
                reps, _ = encoder(images)
                logits = classifier(reps)
                preds = logits.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        valAcc = correct / total
        history["valAcc"].append(valAcc)
        tqdm.write(
            f"Epoch {epoch+1}: trainLoss={avgTrainLoss:.4f}, valAcc={valAcc:.4f}")
        classifier.train()
    return {"bestValAcc": max(history["valAcc"]), "history": history}


# -------------------------
# Dataset wrapper for two views
# -------------------------
class CIFAR10Pair(Dataset):
    """
    CIFAR-10 dataset wrapper that returns two transformed views for contrastive learning.
    """

    def __init__(self, root: str, train: bool, transform):
        self.dataset = datasets.CIFAR10(root=root, train=train, download=True)
        self.transform = TwoCropTransform(transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        x1, x2 = self.transform(img)
        return (x1, x2), label
