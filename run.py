# %%
import torch
from torch import nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
    random_split
)
from torch import optim

import matplotlib.pyplot as plt

from utils.base import LinearRegression, Trainer

plt.style.use("seaborn-v0_8")
plt.rcParams["font.family"] = "monospace"
# %%
# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
BATCH_SIZE = 16
NUM_EPOCHS = 64
SIZE = 200
GENERATOR = torch.Generator().manual_seed(42)

print(f"Device has ben set to: {torch.cuda.get_device_properties(DEVICE).name}")
# %%
X = torch.randn(SIZE, 2, dtype=DTYPE, device='cpu')
y = 2 * X[:, 0] + 3 * X[:, 1] + 5 + torch.randn(SIZE, dtype=DTYPE, device='cpu')

Data = TensorDataset(X, y)
# %%
plt.scatter(X[:, 0].numpy(), y, s=20, edgecolors="b");
plt.scatter(X[:, 1].numpy(), y, s=20, edgecolors="b");
plt.grid(True, alpha = .6);
plt.title("Random Generated Data");
plt.show()
# %%
Model = LinearRegression(in_dims=2).to(DEVICE)

trainData, valData = random_split(Data, (0.8, 0.2), generator=GENERATOR)

trainLoader = DataLoader(trainData, batch_size=BATCH_SIZE, shuffle=True)
valLoader = DataLoader(valData, batch_size=BATCH_SIZE, shuffle=True)

trainer = Trainer(
    Model,
    trainLoader,
    valLoader,
    optimizer=optim.SGD(Model.parameters(), lr=1),
    criterion=nn.L1Loss(reduction='mean'),
    device=DEVICE
)
# %%
train_loss, val_loss = trainer.train(num_epochs=NUM_EPOCHS)

plt.plot(
    train_loss.keys(),
    train_loss.values()
);
plt.ylim(bottom=0)
plt.grid(True, alpha = .6);
plt.title("Training Loss");
plt.show()
# %%

# TODO: Finish implementation with proper graphs, tests and recquired characteristics for the model
# TODO: Create a class for saving performance for given metrics at each epoch and plotting for selected metrics
# %%