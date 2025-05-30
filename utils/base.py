import os

import torch
from torch import (nn, Tensor)
from torch.utils.data import (Dataset, DataLoader)
from torch.nn import Module, Module
import torch.nn.functional as Func

import logging
from logging import handlers

from pathlib import Path

from tqdm import tqdm

from typing import (
    Any,
    Callable,
    List,
    Optional,
    Tuple,
    Union,
)

LIB: Path | str = Path(__file__).parent
ROOT: Path | str = LIB.parent
DATA_DIR: Path | str = ROOT / "data"
TEST_DIR: Path | str = ROOT / "test"
LOG_DIR: Path | str = ROOT / "logs"

class BaseLogger(logging.Logger):
    def __init__(self, name: str, level: int = logging.NOTSET):
        """
        Base Logger class for logging, handling exceptions and configuration.

        :param name: Name of the logger.
        :type name: str
        :param level: Logging level. Default is ``logging.NOTSET``.
        :type level: int
        """
        super().__init__(name, level)
        self._formatter = logging.Formatter(     # >>> 14:54:23 | INFO: BaseLogger @ config.py: Ln 48
            ">>> %(asctime)s | %(levelname)s: %(msg)s -> %(name)s @ %(filename)s: Ln %(lineno)d",
            datefmt="%H:%M:%S",
        )
        self._handler()

    def _handler(self) -> None:
        """
        Set the handler for the logger. This is a "stream" handler that outputs to the console.
        """
        _handler = logging.StreamHandler()
        _handler.setFormatter(self._formatter)

        self.addHandler(_handler)


class FileLogger(BaseLogger):
    def __init__(
            self, name: str,
            filename: str,
            path: str | Path = LOG_DIR,
            level: int = logging.INFO,
            n_backup: int = 0
    ):
        """
        Logger that saves the logs to a set of files.

        :param name: Name of the logger.
        :type name: str
        :param filename: Name of the file to save the logs.
        :type filename: str
        :param path: Path to the directory where the log file will be saved. Default is ``'logs'`` directory.
        :type path: str
        :param level: Logging level. Default is ``logging.INFO``.
        :type level: int
        :param n_backup: Number of backup files to keep. Default is ``0`` (No rollover).
        :type n_backup: int
        """
        self._filename = filename
        self._path = path
        self._n_backup = n_backup
        # Rest of the parameters are passed to the parent class
        super().__init__(name, level)
        
    def _handler(self) -> None:
        """
        Set the handler for the logger.
        """
        if not os.path.exists(self._path):
            os.makedirs(self._path)
        # Set the file handler with rotation
        _file_handler = handlers.RotatingFileHandler(
            filename=Path(self._path) / self._filename, 
            maxBytes=1024,
            backupCount=self._n_backup
        )
        _file_handler.setFormatter(self._formatter)
        
        self.addHandler(_file_handler)

class CustomDataset(Dataset):
    """
    A custom dataset class for handling data and corresponding labels.

    This class inherits from PyTorch's :class:`torch.utils.data.Dataset` and allows the option to apply
    transformations on the data samples.

    :ivar data: List, array, or tensor of data samples.
    :ivar labels: Corresponding labels for the data samples.
    :ivar transform: A callable function or transformation to apply to each data sample.
    """
    def __init__(
        self, 
        data: Union[List[Any], torch.Tensor], 
        labels: Union[List[Any], torch.Tensor], 
        transform: Optional[Callable] = None
    ) -> None:
        """
        Initializes the dataset with data samples and labels, along with an optional transform.

        :param data: List or tensor of data samples.
        :type data: Union[List[Any], torch.Tensor]
        :param labels: Corresponding labels for the data samples.
        :type labels: Union[List[Any], torch.Tensor]
        :param transform: Optional transformation to be applied to the data samples, by default None.
        :type transform: Optional[Callable]
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        :return: Number of samples in the dataset.
        :rtype: int
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        Retrieves a sample and its corresponding label by index.

        :param idx: Index of the sample and label to retrieve.
        :type idx: int
        :return: A tuple containing the data sample and its corresponding label.
        :rtype: Tuple[Any, Any]

        .. note::
            If the ``labels`` attribute is ``None``, only the sample is returned.
        """
        sample = self.data[idx]
        label = self.labels[idx] if self.labels is not None else None

        if self.transform:
            sample = self.transform(sample)

        return (sample, label) if label is not None else (sample, None)

class Trainer:
    """
    Custom trainer class for training and validating a PyTorch model.
    """
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device):
        """
        Initializes the Trainer class.

        :param model: The PyTorch model to be trained.
        :type model: torch.nn.Module
        :param train_loader: DataLoader for the training dataset.
        :type train_loader: torch.utils.data.DataLoader
        :param val_loader: DataLoader for the validation dataset.
        :type val_loader: torch.utils.data.DataLoader
        :param optimizer: Optimizer for the training process (e.g., Adam, SGD).
        :type optimizer: torch.optim.Optimizer
        :param criterion: Loss function for calculating the loss.
        :type criterion: torch.nn.Module
        :param device: Device to run the computations (either 'cpu' or 'cuda').
        :type device: torch.device
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.trainLoss = {}
        self.valLoss = {}

    def __repr__(self) -> str:
        """
        Returns a string representation of the Trainer object,
        including the model type, data loaders, optimizer, criterion, and device.

        :return: String summary of the Trainer.
        :rtype: str
        """
        try:
            train_batches = len(self.train_loader)
        except Exception:
            train_batches = "N/A"
        try:
            val_batches = len(self.val_loader)
        except Exception:
            val_batches = "N/A"
        
        repr_str = (
            f"{' Trainer Summary ':=^50}\n"
            f"Model        : {self.model.__class__.__name__}\n"
            f"  {self.model}\n"
            f"Train Loader : {self.train_loader.__class__.__name__} ({train_batches} batches)\n"
            f"Val Loader   : {self.val_loader.__class__.__name__} ({val_batches} batches)\n"
            f"Optimizer    : {self.optimizer.__class__.__name__}\n"
            f"  {self.optimizer}\n"
            f"Criterion    : {self.criterion}\n"
            f"Device       : {self.device}\n"
            f"{'='*50}\n"
        )
        return repr_str


    def train(self, num_epochs):
        """
        Trains the model for a given number of epochs with a single progress bar.

        :param num_epochs: Number of epochs to train the model.
        :type num_epochs: int
        :return: Training and validation loss dictionaries.
        :rtype: Tuple[dict, dict]
        """
        progress_bar = tqdm(total=num_epochs, desc="Training Progress", position=0, leave=True)

        for epoch in range(1, num_epochs + 1):
            # Set the model to training mode
            self.model.train()
            total_loss = 0

            for batch in self.train_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                predictions = self.model(inputs)
                
                # For Hinge Loss
                # TODO: Redundant and messy, find a more optimal solution 
                if isinstance(self.criterion, Hinge):
                    loss = self.criterion(predictions, targets, self.model.linear.weight)
                else:
                    loss = self.criterion(predictions, targets)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            self.trainLoss[epoch] = avg_loss

            # Validation step
            val_loss = self.validate(epoch)
            self.valLoss[epoch] = val_loss

            progress_bar.set_description(f"Epoch {epoch}/{num_epochs} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")
            progress_bar.update(1)

        progress_bar.close()
        return self.trainLoss, self.valLoss

    def validate(self, epoch):
        """
        Validates the model after each epoch.

        :param epoch: The current epoch number.
        :type epoch: int
        :return: Average validation loss for the epoch.
        :rtype: float
        """
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch in self.val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)

                # For Hinge Loss
                # TODO: Redundant and messy, find a more optimal solution 
                if isinstance(self.criterion, Hinge):
                    val_loss = self.criterion(outputs, targets, self.model.linear.weight)
                else:
                    val_loss = self.criterion(outputs, targets)

                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(self.val_loader)
        return avg_val_loss

class LinearRegression(Module):
    """
    A simple linear regression model implemented with PyTorch.

    :ivar w: Weights for the linear regression model.
    :ivar b: Bias for the linear regression model.
    """
    def __init__(self, in_dims: int, out_dims: int = 1):
        """
        Initializes the LinearRegression model with random weights and bias.

        :param in_dims: Number of input features (dimension of the input).
        :type in_dims: int
        :param out_dims: Number of output features (default is 1 for basic regression).
        :type out_dims: int
        """
        super().__init__()

        self.linear = nn.Linear(in_dims, out_dims, bias=True)

    def forward(self, X: Tensor) -> Tensor:
        """
        Performs the forward pass through the linear regression model.

        :param X: Input tensor of shape (batch_size, in_dims).
        :type X: torch.Tensor
        :return: Predicted output tensor of shape (batch_size, out_dims).
        :rtype: torch.Tensor
        """
        return self.linear(X)

class LogisticRegression(LinearRegression):
    """
    A simple logistic regression model implemented with PyTorch, inheriting from the :class:`LinearRegression` class.

    :ivar w: Weights of the logistic regression model.
    :ivar b: Bias term of the logistic regression model.
    :ivar multinomial: Indicates whether the model is for binary or multinomial classification.
    """
    def __init__(self, in_dims: int, out_dims: int = 1, multinomial: bool = False):
        """
        Initializes the LogisticRegression model with random weights and bias.

        :param in_dims: Number of input features (dimension of the input).
        :type in_dims: int
        :param out_dims: Number of output features. For binary classification, this is usually 1. Default is 1.
        :type out_dims: int
        :param multinomial: Indicates whether the model is for binary or multinomial classification.
        :type multinomial: bool
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

        :param X: Input tensor of shape (batch_size, in_dims).
        :type X: Tensor
        :return: Predicted probabilities:
            - For binary classification, shape is (batch_size, 1), with probabilities for the positive class.
            - For multinomial classification, shape is (batch_size, out_dims), with probabilities over all classes.
        :rtype: Tensor
        """
        logits = super().forward(X)

        if self.multinomial:
            return torch.softmax(logits, dim=1)  # Normalize along the class dimension
        
        return torch.sigmoid(logits)
    
class Hinge(Module):
    """
    Calculates the hinge loss for SVM.

    :param reduction: Specifies the reduction to apply to the output: ``'none' | 'mean' | 'sum'``.
        ``'none'``: no reduction will be applied,
        ``'mean'``: the sum of the output will be divided by the number of
        elements in the output, ``'sum'``: the output will be summed.
        Default: ``'mean'``
    :type reduction: str
    :param is_soft: Whether to use soft-margin SVM. Default is False.
    :type is_soft: bool
    :param C: Regularization parameter (inverse of regularization strength). Used only when ``is_soft`` is True. Default is 1.0.
    :type C: float
    """
    def __init__(self, reduction: str = 'mean', is_soft: bool = False, C: float = 1.0) -> None:
        """
        Initializes the Hinge module.

        :param reduction: Specifies the reduction to apply to the output: ``'none' | 'mean' | 'sum'``.
            ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.
            Default: ``'mean'``
        :type reduction: str
        :param is_soft: Whether to use soft-margin SVM. Default is False.
        :type is_soft: bool
        :param C: Regularization parameter (inverse of regularization strength). Used only when ``is_soft`` is True. Default is 1.0.
        :type C: float
        """
        super().__init__()
        self.reduction = reduction
        self.is_soft = is_soft
        self.C = C

    def __repr__(self) -> str:
        """
        Returns a string representation of the Hinge loss instance.

        This representation includes the reduction method used, whether soft-margin SVM is enabled,
        and the regularization parameter C.

        :return: A formatted string representing the current state of the Hinge loss instance.
        :rtype: str
        """
        return f"Hinge(reduction: {self.reduction}, is_soft: {self.is_soft}, C: {self.C})"

    def forward(self, output: Tensor, target: Tensor, weights: Optional[Tensor]) -> Tensor:
        """
        Calculates the hinge loss.

        :param output: The output from the model.
        :type output: Tensor
        :param target: The target values (should be 1 or -1).
        :type target: Tensor
        :param weights: The model weights (required for soft-margin SVM).
        :type weights: Optional[Tensor]
        :return: The calculated hinge loss.
        :rtype: Tensor
        """
        loss = Func.relu(1 - target * output.squeeze())

        if self.reduction == 'mean':
            hinge_loss = loss.mean()
        elif self.reduction == 'sum':
            hinge_loss = loss.sum()
        else:
            hinge_loss = loss    
        
        if self.is_soft:
            if weights is None:
                raise ValueError("Weight must be provided for soft-margin classification!")            
            # Calculate regularization term
            reg_loss = 0.5 * torch.linalg.norm(weights) ** 2
            total_loss = reg_loss + self.C * hinge_loss

            return total_loss
        
        return hinge_loss