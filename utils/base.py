import torch
import numpy as np
from torch import nn
from torch.utils.data import (Dataset, DataLoader)
from torch.optim import Optimizer
from torch.nn import Module, Module as LossFunction
from torch import Tensor

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from typing import (
    Any,
    Callable,
    List,
    Optional,
    Tuple,
    Union,
    Dict
)

class CustomDataset(Dataset):
    """
    A custom dataset class for handling data and corresponding labels. 
    This class inherits from PyTorch's `Dataset` and allows the option to apply 
    transformations on the data samples.
    
    Attributes:
        data (Union[List[Any], np.ndarray]): List or array of data samples.
        labels (Union[List[Any], np.ndarray]): Corresponding labels for the data samples.
        transform (Optional[Callable]): A callable function or transformation to apply to each data sample.
    
    Methods:
        __len__(): Returns the total number of samples in the dataset.
        __getitem__(idx: int): Retrieves a sample and its label by index, applying transformation if available.
    """

    def __init__(
        self, 
        data: Union[List[Any], torch.Tensor], 
        labels: Union[List[Any], torch.Tensor], 
        transform: Optional[Callable] = None
    ) -> None:
        """
        Initializes the dataset with data samples and labels, along with an optional transform.

        Args:
            data (Union[List[Any], torch.Tensor]): List or tensor of data samples.
            labels (Union[List[Any], torch.Tensor]): Corresponding labels for the data samples.
            transform (Optional[Callable]): Optional transformation to be applied to the data samples.
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        Retrieves a sample and its corresponding label by index.

        Args:
            idx (int): Index of the sample and label to retrieve.

        Returns:
            Tuple[Any, Any]: A tuple containing the data sample and its corresponding label.
        """
        sample = self.data[idx]
        label = self.labels[idx]

        if self.labels is not None:
            label = self.labels[idx]
            return sample, label
			
        return sample

class Trainer:
    """
    Custom trainer class for training and validating a PyTorch model.

    Attributes:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters.
        criterion (torch.nn.Module): Loss function to be used during training and validation.
        device (torch.device): The device on which the model and data are placed (e.g., 'cpu', 'cuda').
        trainLoss (Dict[int, float]): Dictionary that stores the average training loss for each epoch.
        valLoss (Dict[int, float]): Dictionary that stores the average validation loss for each epoch.
    """
    def __init__(
        self, 
        model: nn.Module, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        optimizer: Optimizer, 
        criterion: nn.Module, 
        device: Union[torch.device, str]
    ) -> None:
        """
        Initializes the Trainer class.

        Args:
            model (torch.nn.Module): The PyTorch model to be trained.
            train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
            val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
            optimizer (torch.optim.Optimizer): Optimizer for the training process (e.g., Adam, SGD).
            criterion (torch.nn.Module): Loss function (e.g., CrossEntropyLoss) for calculating the loss.
            device (torch.device or str): Device to run the computations (either 'cpu' or 'cuda').
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.trainLoss: Dict[int, float] = {}
        self.valLoss: Dict[int, float] = {}

    def train(self, num_epochs: int) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        Trains the model for a given number of epochs.

        Args:
            num_epochs (int): The number of epochs to train the model.
        
        Returns:
            (trainLoss, valLoss):
            A tuple containing two dictionaries of both losses for each epoch
        """
        for epoch in range(num_epochs):
            self.model.train()  # Set the model to training mode
            total_loss = 0
            
            for batch in self.train_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                predictions = self.model(inputs)
                loss = self.criterion(predictions, targets)
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()
                
                print(f'\tEpoch {epoch + 1} | Weights: {self.model.w.data}')
                print(f'\tEpoch {epoch + 1} | Bias: {self.model.b.data}')
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(self.train_loader)
            print(f'Epoch {epoch + 1}/{num_epochs} | Loss: {avg_loss:.4f}')

            self.trainLoss[epoch] = avg_loss

            # Call the validation step after each epoch
            self.validate(epoch)

        return self.trainLoss, self.valLoss

    def validate(self, epoch: int) -> None:
        """
        Validates the model after each epoch.

        Args:
            epoch (int): The current epoch number to store validation loss.

        Returns:
            None
        """
        self.model.eval()  # Set the model to evaluation mode
        total_val_loss = 0
        
        with torch.no_grad():  # No need to compute gradients during validation
            for batch in self.val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                val_loss = self.criterion(outputs, targets)
                
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(self.val_loader)  # Calculate the average validation loss
        print(f'\t| Validation Loss: {avg_val_loss:.4f}\n')

        self.valLoss[epoch] = avg_val_loss

# TODO: `LossVisualizer` breaks the trained weights. Resolve the issue.
class LossVisualizer:
    """
    A class to visualize the loss surface for a linear regression model with two features.
    
    Attributes:
        model (torch.nn.Module): A trained linear regression model.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        criterion (torch.nn.Module): Loss function (e.g., Mean Squared Error) to visualize.
        w1_range (tuple): Range of values for the first weight (w1).
        w2_range (tuple): Range of values for the second weight (w2).
        device (torch.device or str): Device where the computations will be performed ('cpu' or 'cuda').
    """

    def __init__(
            self,
            model,
            train_loader,
            criterion,
            w1_range=(-10, 10), w2_range=(-10, 10),
            resolution=50,
            device='cpu'
    ):
        """
        Initializes the LossVisualizer class.
        
        Args:
            model (torch.nn.Module): A trained linear regression model.
            train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
            criterion (torch.nn.Module): Loss function (e.g., Mean Squared Error).
            w1_range (tuple): Range for the first weight (w1).
            w2_range (tuple): Range for the second weight (w2).
            resolution (int): Resolution of the grid for visualizing the loss surface.
            device (str or torch.device): Device where the computations will be performed ('cpu' or 'cuda').
        """
        self.model = model.to(device)  # Ensure the model is on the correct device
        self.train_loader = train_loader
        self.criterion = criterion
        self.w1_range = w1_range
        self.w2_range = w2_range
        self.resolution = resolution
        self.device = device  # Ensure all calculations will be performed on this device

    def calculate_loss(self, w1: float, w2: float) -> float:
        """
        Calculate the loss for a given pair of weights (w1, w2).
        
        Args:
            w1 (float): The value of the first weight.
            w2 (float): The value of the second weight.
        
        Returns:
            float: The calculated loss.
        """
        # Temporarily set the weights in the model
        with torch.no_grad():
            self.model.w[0] = torch.tensor(w1, dtype=torch.float32).to(self.device)
            self.model.w[1] = torch.tensor(w2, dtype=torch.float32).to(self.device)

        total_loss = 0
        num_samples = 0

        for batch in self.train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            num_samples += inputs.size(0)

        return total_loss / num_samples     # Average Loss

    def plot_loss_surface(self) -> None:
        """
        Plots the 3D loss surface using w1 and w2 as axes.
        
        Returns:
            None
        """
        # Create a grid of values for w1 and w2
        w1_values = np.linspace(self.w1_range[0], self.w1_range[1], self.resolution)
        w2_values = np.linspace(self.w2_range[0], self.w2_range[1], self.resolution)
        w1_grid, w2_grid = np.meshgrid(w1_values, w2_values)
        
        # Calculate the loss for each combination of w1 and w2
        loss_grid = np.zeros_like(w1_grid)
        for i in range(self.resolution):
            for j in range(self.resolution):
                loss_grid[i, j] = self.calculate_loss(w1_grid[i, j], w2_grid[i, j])

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(w1_grid, w2_grid, loss_grid, cmap='viridis')

        ax.set_xlabel('Weight $W_1$', labelpad=10)
        ax.set_ylabel('Weight $W_2$', labelpad=10)
        ax.set_title('Loss Surface for Linear Regression')

        plt.show()

class LinearRegression(Module):
    """
    A simple linear regression model implemented with PyTorch.

    Attributes:
        w (Tensor): Weights for the linear regression model.
        b (Tensor): Bias for the linear regression model.
    
    Methods:
        forward(X: Tensor) -> Tensor: Performs a forward pass (predicts the output for input X).
    """

    def __init__(self, in_dims: int, out_dims: int = 1):
        """
        Initializes the LinearRegression model with random weights and bias.

        Args:
            in_dims (int): Number of input features (dimension of the input).
            out_dims (int): Number of output features (usually 1 for basic regression).
        """
        super(LinearRegression, self).__init__()

        self.w = nn.Parameter(torch.randn(in_dims, out_dims).squeeze())
        self.b = nn.Parameter(torch.randn(out_dims))

    def forward(self, X: Tensor) -> Tensor:
        """
        Performs the forward pass through the linear regression model.

        Args:
        X (Tensor): Input tensor of shape (batch_size, in_dims)

        Returns:
            Tensor: Predicted output of shape (batch_size, out_dims)
        """
        return torch.matmul(X, self.w) + self.b

class LogisticRegression(LinearRegression):
    """
    A simple logistic regression model implemented with PyTorch, inheriting from `LinearRegression` class

    Attributes:
        w (Tensor): Weights of the logistic regression model.
        b (Tensor): Bias term of the logistic regression model.
    
    Methods:
        forward(X: Tensor) -> Tensor: Performs a forward pass and outputs probabilities.
    """
    
    def __init__(self, in_dims: int, out_dims: int = 1):
        """
        Initializes the LogisticRegression model with random weights and bias.

        Args:
            in_dims (int): Number of input features (dimension of the input).
            out_dims (int): Number of output features (usually 1 for binary classification).
        """
        super().__init__(in_dims, out_dims)

    def forward(self, X: Tensor) -> Tensor:
        """
        Performs the forward pass through the logistic regression model.

        Args:
            X (Tensor): Input tensor of shape (batch_size, in_dims).

        Returns:
            Tensor: Predicted probabilities of shape (batch_size, out_dims), where each element 
            represents the probability of the positive class for binary classification.
        """
        return torch.sigmoid(super().forward(X))