import torch
from torch import (optim, nn, Tensor)

from torch.utils.data import (
    DataLoader,
    TensorDataset,
    random_split
)

import numpy as np

# User-defined classes
from utils.base import (
    LinearSVM,
    HingeLoss,
    Trainer
)

# User-defined functions
from utils.preprocess import clusters, onehot

# Plotting
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")
plt.rcParams["font.family"] = "monospace"

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
BATCH_SIZE = 32
NUM_EPOCHS = 50
SIZE = 100
GENERATOR = torch.Generator().manual_seed(42)

print(f"Device has ben set to: {torch.cuda.get_device_properties(DEVICE).name}")

# ALTERNATIVE Data generation (ensures perfect separability):
X1 = torch.randn(SIZE, 2, generator=GENERATOR) + torch.tensor([-3, -3])
X2 = torch.randn(SIZE, 2, generator=GENERATOR) + torch.tensor([3, 3])
X = torch.cat([X1, X2], dim=0)
y = torch.cat([torch.ones(SIZE), -torch.ones(SIZE)], dim=0) # MORE ROBUST LABELING

#Original clustering
#X, y = clusters(SIZE, means=[(-3, -3), (3, 3)], stds=[0.9, 1.2])

#y[y == 0] = -1.0
#y[y == 1] = 1.0

print(f"New label values in y: \n{y}\nData Type: \n{y.dtype}")
print(f"Unique data on the dataset: \n{torch.unique(y)}")

# Standardize input features (Mean 0, Unit Variance)
mean = X.mean(dim=0, keepdim=True)  # Calculate mean for each feature
std = X.std(dim=0, keepdim=True)    # Calculate standard deviation for each feature
X = (X - mean) / (std + 1e-8)         # Standardize (add a small epsilon to prevent division by zero)

class LinearRegression(nn.Module):
    """
    Linear Regression model.
    """
    def __init__(self, in_dims: int) -> None:
        """
        Initializes the LinearRegression model.
        """
        super().__init__()
        self.linear = nn.Linear(in_dims, 1)

    def forward(self, X: torch.Tensor) -> Tensor:
        """
        Performs the forward pass through the linear SVM model.
        """
        return self.linear(X)

class LinearSVM(LinearRegression):
    """
    Linear Support Vector Machine (SVM) model.
    """
    def __init__(self, in_dims: int) -> None:
        """
        Initializes the LinearSVM model.
        """
        super().__init__(in_dims)
        #Weight initialization
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, X: torch.Tensor) -> Tensor:
        """
        Performs the forward pass through the linear SVM model.
        """
        return super().forward(X)

class HingeLoss(nn.Module):
    """
    Calculates the hinge loss for SVM.
    """
    def __init__(self, reduction: str = 'mean', is_soft: bool = False, C: float = 1.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.is_soft = is_soft
        self.C = C

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        """
        Calculates the hinge loss.
        """
        loss = torch.mean(nn.functional.relu(1 - target * output.squeeze()))
        if self.is_soft:

            reg_loss = 0.0
            for param in self.parameters():
                reg_loss += torch.linalg.norm(param)
            
            loss += (self.C / 2) * reg_loss

        if self.reduction == 'sum':
            loss = torch.sum(nn.functional.relu(1 - target * output.squeeze()))
        elif self.reduction == 'none':
            loss = nn.functional.relu(1 - target * output.squeeze())
        return loss


# Model and Training
model = LinearSVM(in_dims=2).to(DEVICE)
Data = TensorDataset(X, y)
trainData, valData = random_split(Data, (0.8, 0.2), generator=GENERATOR)
trainLoader = DataLoader(trainData, batch_size=BATCH_SIZE, generator=GENERATOR, shuffle=True)
valLoader = DataLoader(valData, batch_size=BATCH_SIZE, generator=GENERATOR, shuffle=True)

#Set Soft-margin to True
trainer = Trainer(
    model,
    trainLoader,
    valLoader,
    optimizer=optim.Adam(model.parameters(), lr=.1),
    criterion=HingeLoss(reduction='mean', is_soft=True, C=1),
    device=DEVICE
)

for X_batch, y_batch in trainLoader:
    print("Shape for X_batch is", X_batch.shape)
    break

train_loss, val_loss = trainer.train(num_epochs=NUM_EPOCHS)

print(f"Predicted model Parameters:", "Weights: {}".format(model.linear.weight), "Bias: {}".format(model.linear.bias), sep="\n")

model.to('cpu')

# Extract weights and bias
w = model.linear.weight.detach().numpy()
b = model.linear.bias.detach().numpy()

X_np = X.numpy()
y_np = y.numpy()

x_min, x_max = X_np[:, 0].min() - 1, X_np[:, 0].max() + 1
y_min, y_max = X_np[:, 1].min() - 1, X_np[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_tensor = torch.from_numpy(grid_points).float()

# Standardization of Grid Tensor
mean = X.mean(dim=0, keepdim=True)
std = X.std(dim=0, keepdim=True)
grid_tensor = (grid_tensor - mean) / (std + 1e-8)

# Inference
model.eval()

Z = model(grid_tensor)

Z = Z.detach().numpy()
Z = Z.reshape(xx.shape)

# Plotting
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.5)  # Decision regions
plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np, cmap=plt.cm.RdBu, edgecolors='k')

# Hyperplane
x_plot = np.linspace(x_min, x_max, 100)
y_plot = (-w[0, 0] * x_plot - b) / w[0, 1]
plt.plot(x_plot, y_plot, 'k-', label='Hyperplane')

# Margins
margin = 1 / np.linalg.norm(w)
y_upper = (-w[0, 0] * x_plot - b + 1) / w[0, 1]
y_lower = (-w[0, 0] * x_plot - b - 1) / w[0, 1]
plt.plot(x_plot, y_upper, 'k--', label='Margins')
plt.plot(x_plot, y_lower, 'k--')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Decision Boundary and Margins')
plt.legend()
plt.show()