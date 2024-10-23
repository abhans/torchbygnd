# %%
import torch
from torch import nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
    random_split
)
from torch import optim

import numpy as np
import matplotlib.pyplot as plt

from utils.base import (LinearRegression, Trainer, LossVisualizer)

plt.style.use("seaborn-v0_8")
plt.rcParams["font.family"] = "monospace"
# %%
# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
BATCH_SIZE = 32
NUM_EPOCHS = 50
SIZE = 200
GENERATOR = torch.Generator().manual_seed(42)

print(f"Device has ben set to: {torch.cuda.get_device_properties(DEVICE).name}")
# %%
X = torch.randn(SIZE, 2, dtype=DTYPE, device='cpu')
y = 2 * X[:, 0] + 3 * X[:, 1] + 5 + torch.randn(SIZE, dtype=DTYPE, device='cpu')

Data = TensorDataset(X, y)
# %%
# plt.scatter(X[:, 0].numpy(), y.numpy(), s=20, edgecolors="b");
# plt.scatter(X[:, 1].numpy(), y.numpy(), s=20, edgecolors="b");
# plt.grid(True, alpha = .6);
# plt.title("Random Generated Data");
# plt.show()
# %%
Model = LinearRegression(in_dims=2).to(DEVICE)
# Model = nn.Linear(in_features=2, out_features=1, bias=True)

trainData, valData = random_split(Data, (0.8, 0.2), generator=GENERATOR)

trainLoader = DataLoader(trainData, batch_size=BATCH_SIZE, shuffle=True)
valLoader = DataLoader(valData, batch_size=BATCH_SIZE, shuffle=True)

trainer = Trainer(
    Model,
    trainLoader,
    valLoader,
    optimizer=optim.SGD(Model.parameters(), lr=.1),
    criterion=nn.L1Loss(reduction='mean'),
    device=DEVICE
)
# %%
train_loss, val_loss = trainer.train(num_epochs=NUM_EPOCHS)

# plt.plot(
#     train_loss.keys(),
#     train_loss.values()
# );
# plt.ylim(bottom=0)
# plt.grid(True, alpha = .6);
# plt.title("Training Loss");
# plt.show()
# %%
# TODO: `LossVisualizer` breaks the trained weights. Resolve the issue.
# Visualizer = LossVisualizer(
#     Model,
#     trainLoader,
#     criterion=nn.L1Loss(reduction='mean'),
#     w1_range=(-10, 10), w2_range=(-10, 10)
# )
# Visualizer.plot_loss_surface()
# %%
print(
    f"Trained Weights: {Model.w.data}",
    f"Trained Bias: {Model.b.data}",
    sep="\n"
)

T = np.linspace(X.min(), X.max(), SIZE, dtype=np.float32).reshape(SIZE, 1)
T = torch.tensor(np.concatenate([T, T], axis=1), device=DEVICE)

with torch.no_grad():
    yT = Model(T)

# print(f"\nGenerated T:\n{T}")
# print(f"\nPredictions:\n{yT}")

plt.scatter(X[:, 0].numpy(), y.numpy(), s=20, edgecolors="b");
plt.scatter(X[:, 1].numpy(), y.numpy(), s=20, edgecolors="b");
# Predicted Linear Model
plt.plot(T[:, 0].cpu().numpy(), yT.cpu().numpy(), color="black", alpha=.7, linestyle='--', label="Predictions");
plt.grid(True, alpha = .6);
plt.title("Trained Model");
plt.xlabel("$W_1, W_2$");
plt.ylabel("Target ($y_{pred}$)")
plt.legend(loc='best');
plt.show()