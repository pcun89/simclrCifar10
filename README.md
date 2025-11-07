# SimCLR: Self-Supervised Representation Learning on CIFAR-10 🧠

This project implements a **SimCLR-style self-supervised learning framework** using PyTorch.  
It learns image representations from unlabeled data and evaluates them through a **linear classification** task on CIFAR-10.

## 🚀 Overview

SimCLR (Simple Framework for Contrastive Learning of Visual Representations) learns visual features by **maximizing agreement between differently augmented views of the same image**.  
Once pretrained, the learned encoder is evaluated using a **linear classifier** — a common benchmark for representation learning quality.

---

## 🧩 Features

- Custom **ResNet18 encoder** + projection head
- **NT-Xent loss** (contrastive loss function)
- **Data augmentation pipeline** with random crops, color jitter, Gaussian blur, etc.
- **Two-view dataset wrapper** for contrastive training
- **Linear evaluation protocol** on CIFAR-10
- Checkpoint saving/loading for pretrained encoders
- GPU/CPU compatible with command-line flags
- Modular & well-documented single-file structure

---

## 🧠 Architecture

CIFAR-10 image
├── Random Crop, Flip, Blur, etc. ───▶ view1
└── Random Crop, Flip, Blur, etc. ───▶ view2
Both views → ResNet18 Encoder → Projection MLP → NT-Xent Loss


After pretraining:


Frozen Encoder → Linear Classifier → Evaluate on CIFAR-10 Labels


---

## 🛠️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/simclr-cifar10.git
cd simclr-cifar10
pip install torch torchvision tqdm

🧑‍💻 Usage
1️⃣ Pretrain the encoder
python simclr_cifar10.py --epochs_pretrain 50 --batch_size 256 --device cuda


This will:

Train the model using contrastive learning.

Save the best encoder weights to simclr_encoder_best.pt.

2️⃣ Run linear evaluation

After pretraining completes, the script automatically trains a linear classifier on frozen features:

Evaluates accuracy on the CIFAR-10 test set.

Saves final model as simclr_full_checkpoint.pt.

🧮 Command-line Arguments
Argument	Default	Description
--data_root	./data	Path to CIFAR-10 data
--batch_size	256	Training batch size
--epochs_pretrain	20	Number of epochs for contrastive pretraining
--epochs_linear	10	Number of epochs for linear evaluation
--lr_pretrain	1e-3	Learning rate for encoder
--lr_linear	1e-2	Learning rate for linear classifier
--projection_dim	128	Dimension of projection head
--device	cuda (if available)	Device for training

Example:

python simclr_cifar10.py --epochs_pretrain 100 --batch_size 512 --device cuda

🧠 Results Example
Stage	Metric	Value
Pretraining	Avg Contrastive Loss	~0.20 after 50 epochs
Linear Evaluation	Test Accuracy	~70–75% (ResNet18 + CIFAR10)

Results may vary based on batch size, epochs, and GPU used.

🧱 Data Structures Used

torch.Tensor → For all image batches, representations, projections

torch.utils.data.Dataset / DataLoader → Handles augmentation and batching

nn.Module → Model structure for encoder, projection, classifier

dict & list → Store histories, metrics, checkpoints

Sequential → For MLP projection head

🕒 Complexity Analysis

Time Complexity: O(N * E * I)

N: number of samples

E: encoder forward pass cost

I: pretraining epochs

Space Complexity: O(B * A + M)

B: batch size

A: activation size

M: model parameters

🐳 Running in Docker
Build the image
docker build -t simclr-cifar10 .

Run the container
docker run --gpus all -it --rm -v $(pwd):/app simclr-cifar10 \
python simclr_cifar10.py --epochs_pretrain 20 --epochs_linear 10 --device cuda


If you don’t have a GPU:

docker run -it --rm -v $(pwd):/app simclr-cifar10 \
python simclr_cifar10.py --device cpu

📈 Possible Extensions

Use ResNet50 or Vision Transformer backbone

Add Optuna for hyperparameter tuning

Add t-SNE / UMAP visualization of learned embeddings

Support multi-GPU distributed training

Integrate with Weights & Biases for experiment tracking