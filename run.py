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
import seaborn as sns

from utils.base import LinearRegression, Trainer

plt.style.use("seaborn-v0_8")
plt.rcParams["font.family"] = "monospace"
# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
BATCH_SIZE = 5

print(f"Device has ben set to: {torch.cuda.get_device_properties(DEVICE)}")
# %%
X = torch.randn(100, dtype=DTYPE, device='cpu')
y = 2 * X + 8 + torch.randn(100, dtype=DTYPE, device='cpu')

Data = TensorDataset(X, y)
# %%
# plt.scatter(X, y);
# plt.grid(True, alpha = .6);
# plt.title("Random Generated Data");
# plt.show()
#%%
Model = LinearRegression(5, 1).to(DEVICE)

trainData, valData = random_split(Data, [.8, .2])

trainLoader = DataLoader(trainData, batch_size=BATCH_SIZE, shuffle=True)
valLoader = DataLoader(valData, batch_size=BATCH_SIZE, shuffle=True)
# %%
trainer = Trainer(
    Model,
    trainLoader,
    valLoader,
    optimizer=optim.SGD(Model.parameters(), lr=.1),
    criterion=nn.MSELoss(reduction='mean'),
    device=DEVICE
)

trainer.train(num_epochs=24)
# TODO: Finish implementation with proper graphs, tests and recquired characteristics for the model
# TODO: Create a class for saving performance for given metrics at each epoch and plotting for selected metrics