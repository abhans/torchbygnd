import torch
from tqdm import tqdm
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

    Attributes
    ----------
    data : Union[List[Any], np.ndarray, torch.Tensor]
        List, array, or tensor of data samples.
    labels : Union[List[Any], np.ndarray, torch.Tensor]
        Corresponding labels for the data samples.
    transform : Optional[Callable]
        A callable function or transformation to apply to each data sample.

    Methods
    -------
    __len__()
        Returns the total number of samples in the dataset.
    __getitem__(idx: int)
        Retrieves a sample and its label by index, applying transformation if available.
    """

    def __init__(
        self, 
        data: Union[List[Any], torch.Tensor], 
        labels: Union[List[Any], torch.Tensor], 
        transform: Optional[Callable] = None
    ) -> None:
        """
        Initializes the dataset with data samples and labels, along with an optional transform.

        Parameters
        ----------
        data : Union[List[Any], torch.Tensor]
            List or tensor of data samples.
        labels : Union[List[Any], torch.Tensor]
            Corresponding labels for the data samples.
        transform : Optional[Callable], optional
            Optional transformation to be applied to the data samples, by default None.
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        Retrieves a sample and its corresponding label by index.

        Parameters
        ----------
        idx : int
            Index of the sample and label to retrieve.

        Returns
        -------
        Tuple[Any, Any]
            A tuple containing the data sample and its corresponding label.

        Notes
        -----
        If the `labels` attribute is `None`, only the sample is returned.
        """
        sample = self.data[idx]
        label = self.labels[idx] if self.labels is not None else None

        if self.transform:
            sample = self.transform(sample)

        return (sample, label) if label is not None else sample

class Trainer:
    """
    Custom trainer class for training and validating a PyTorch model.

    Attributes
    ----------
    model : torch.nn.Module
        The model to be trained.
    train_loader : torch.utils.data.DataLoader
        DataLoader for training data.
    val_loader : torch.utils.data.DataLoader
        DataLoader for validation data.
    optimizer : torch.optim.Optimizer
        Optimizer for updating the model parameters.
    criterion : torch.nn.Module
        Loss function to be used during training and validation.
    device : torch.device
        The device on which the model and data are placed (e.g., 'cpu', 'cuda').
    trainLoss : dict[int, float]
        Dictionary that stores the average training loss for each epoch.
    valLoss : dict[int, float]
        Dictionary that stores the average validation loss for each epoch.
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

        Parameters
        ----------
        model : torch.nn.Module
            The PyTorch model to be trained.
        train_loader : torch.utils.data.DataLoader
            DataLoader for the training dataset.
        val_loader : torch.utils.data.DataLoader
            DataLoader for the validation dataset.
        optimizer : torch.optim.Optimizer
            Optimizer for the training process (e.g., Adam, SGD).
        criterion : torch.nn.Module
            Loss function (e.g., CrossEntropyLoss) for calculating the loss.
        device : torch.device or str
            Device to run the computations (either 'cpu' or 'cuda').
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

        Parameters
        ----------
        num_epochs : int
            The number of epochs to train the model.

        Returns
        -------
        tuple[dict[int, float], dict[int, float]]
            A tuple containing two dictionaries: training loss and validation loss for each epoch.
        """
        for epoch in range(num_epochs):
            self.model.train()  # Set the model to training mode
            total_loss = 0

            for batch in tqdm(self.train_loader, desc = f'Epoch {epoch + 1} / {num_epochs}\r'):
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                predictions = self.model(inputs)

                loss = self.criterion(predictions, targets)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            print(f'Loss: {avg_loss:.4f}')

            self.trainLoss[epoch] = avg_loss

            # Call the validation step after each epoch
            self.validate(epoch)

        return self.trainLoss, self.valLoss

    def validate(self, epoch: int) -> None:
        """
        Validates the model after each epoch.

        Parameters
        ----------
        epoch : int
            The current epoch number to store validation loss.

        Returns
        -------
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
                val_loss = self.criterion(outputs.squeeze(), targets)

                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(self.val_loader)  # Calculate the average validation loss
        print(f'\t| Validation Loss: {avg_val_loss:.4f}\n')

        self.valLoss[epoch] = avg_val_loss

class LinearRegression(Module):
    """
    A simple linear regression model implemented with PyTorch.

    Attributes
    ----------
    w : torch.Tensor
        Weights for the linear regression model.
    b : torch.Tensor
        Bias for the linear regression model.

    Methods
    -------
    forward(X: torch.Tensor) -> torch.Tensor
        Performs a forward pass (predicts the output for input X).
    """

    def __init__(self, in_dims: int, out_dims: int = 1):
        """
        Initializes the LinearRegression model with random weights and bias.

        Parameters
        ----------
        in_dims : int
            Number of input features (dimension of the input).
        out_dims : int, optional
            Number of output features (default is 1 for basic regression).
        """
        super(LinearRegression, self).__init__()

        self.linear = nn.Linear(in_dims, out_dims, bias=True)

    def forward(self, X: Tensor) -> Tensor:
        """
        Performs the forward pass through the linear regression model.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (batch_size, in_dims).

        Returns
        -------
        torch.Tensor
            Predicted output tensor of shape (batch_size, out_dims).
        """
        return self.linear(X)

class LogisticRegression(LinearRegression):
    """
    A simple logistic regression model implemented with PyTorch, inheriting from the `LinearRegression` class.

    Attributes
    ----------
    w : Tensor
        Weights of the logistic regression model.
    b : Tensor
        Bias term of the logistic regression model.
    multinomial : bool
        Indicates whether the model is for binary or multinomial classification.

    Methods
    -------
    forward(X: Tensor) -> Tensor
        Performs a forward pass and outputs probabilities.
    """ 
    def __init__(self, in_dims: int, out_dims: int = 1, multinomial: bool = False):
        """
        Initializes the LogisticRegression model with random weights and bias.

        Parameters
        ----------
        in_dims : int
            Number of input features (dimension of the input).
        out_dims : int, optional
            Number of output features. For binary classification, this is usually 1. Default is 1.
        multinomial : bool
            Indicates whether the model is for binary or multinomial classification.
        """
        if multinomial and out_dims < 2:
            raise ValueError("For multinomial classification, out_dims must be at least 2.")
        if not multinomial and out_dims != 1:
            raise ValueError("For binary classification, out_dims must be 1.")
        
        self.multinomial = multinomial
        super().__init__(in_dims, out_dims)

    def forward(self, X: Tensor) -> Tensor:
        """
        Performs the forward pass through the logistic regression model.

        Parameters
        ----------
        X : Tensor
            Input tensor of shape (batch_size, in_dims).

        Returns
        -------
        Tensor
            Predicted probabilities:
            - For binary classification, shape is (batch_size, 1), with probabilities for the positive class.
            - For multinomial classification, shape is (batch_size, out_dims), with probabilities over all classes.
        """
        logits = super().forward(X)

        if self.multinomial:
            return torch.softmax(logits, dim=1)  # Normalize along the class dimension
        
        return torch.sigmoid(logits)