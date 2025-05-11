#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: run.py
Author: Ahmet B. SERCE
Date: 2025-02-17
Version: 0.1a
Description: .py file for basic operations and prototyping
"""
import torch
import logging
from torch import (optim)

from torch.utils.data import (
    DataLoader,
    TensorDataset,
    random_split
)

import numpy as np

# User-defined classes
from utils.base import (
    # BaseLogger,
    FileLogger,
    LinearSVM,
    Hinge,
    Trainer
)

# User-defined functions
from utils.preprocess import clusters

# Plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import colormaps

plt.style.use("seaborn-v0_8")
plt.rcParams["font.family"] = "monospace"

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE: torch.dtype = torch.float32
BATCH_SIZE: int = 32
NUM_EPOCHS: int = 20
SIZE: int = 100
GENERATOR = torch.Generator().manual_seed(42)

logger = FileLogger(
    "RunLogger",
    filename="hinge.log",
    level=logging.INFO
)

logger.info(f"Device has ben set to: {torch.cuda.get_device_properties(DEVICE).name}")

X, y = clusters(SIZE, stds=[1.2, 1], dtype=DTYPE, generator=GENERATOR)

data = TensorDataset(X, y)

trainData, valData = random_split(data, (0.8, 0.2), generator=GENERATOR)

trainLoader = DataLoader(trainData, batch_size=BATCH_SIZE, generator=GENERATOR, shuffle=True)
valLoader = DataLoader(valData, batch_size=BATCH_SIZE, generator=GENERATOR, shuffle=True)

model = LinearSVM(in_dims=2).to(DEVICE)
criteria = Hinge(reduction='mean', is_soft=False)

logger.info(f"Model: {model}")
logger.info(f"Loss: {criteria}")

trainer = Trainer(
    model,
    trainLoader,
    valLoader,
    optimizer=optim.SGD(model.parameters(), lr=.1),
    criterion=criteria,        # Soft Margin SVM
    device=DEVICE
)

train_loss, val_loss = trainer.train(num_epochs=NUM_EPOCHS)

logger.info(f"Train Loss: {train_loss[NUM_EPOCHS]}")
logger.info(f"Validation Loss: {val_loss[NUM_EPOCHS]}")

logger.info(f"Weights: {model.linear.weight}")
logger.info(f"Biases: {model.linear.bias}")