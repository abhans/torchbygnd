import torch
from torch import Tensor

# Function Definitions
def categorical(y: Tensor, num_classes: int) -> Tensor:
    """
    Converts integer labels to one-hot encoded format.

    Parameters
    ----------
    y : Tensor
        A tensor of integer labels of shape ``(N,)``, where each element 
        is an index representing a class label. All labels must satisfy 
        the condition ``0 <= label < num_classes``.
    num_classes : int
        The number of distinct classes.

    Returns
    -------
    Tensor
        A one-hot encoded tensor of shape ``(N, num_classes)``, where 
        each row corresponds to a one-hot encoded vector for each label 
        in ``y``.

    Raises
    ------
    ValueError
        If any label in ``y`` is outside the range ``[0, num_classes - 1]``.
    """
    if (y >= num_classes).any() or (y < 0).any():
        raise ValueError("Labels in `y` should be in the range [0, num_classes - 1].")

    return torch.eye(num_classes, dtype=torch.float32)[y]

def clusters(
        size: int,
        std0: float = 1.0,
        mean0: tuple = (-3, -3),
        std1: float = 1.0,
        mean1: tuple = (3, 3),
        dtype=torch.float32,
        generator=None
):
    """
    Generate two distinct clusters of data for binary classification tasks.

    Parameters
    ----------
    size : int
        Number of points per cluster.
    std0 : float, optional
        Standard deviation of the cluster for ``y=0``. Default is ``1.0``.
    mean0 : tuple, optional
        Mean (center) of the cluster for ``y=0``. Default is ``(-3, -3)``.
    std1 : float, optional
        Standard deviation of the cluster for ``y=1``. Default is ``1.0``.
    mean1 : tuple, optional
        Mean (center) of the cluster for ``y=1``. Default is ``(3, 3)``.
    dtype : torch.dtype, optional
        Data type of the output tensors. Default is ``torch.float32``.
    generator : torch.Generator, optional
        Random generator for reproducibility. Default is ``None``.

    Returns
    -------
    X : torch.Tensor
        Feature matrix of shape ``(2 * size, 2)`` containing the generated data points.
    y : torch.Tensor
        Binary labels of shape ``(2 * size,)`` corresponding to the data points.

    Examples
    --------
    >>> X, y = clusters(size=100, std0=0.5, mean0=(-2, -2), std1=0.5, mean1=(2, 2))
    >>> X.shape
    torch.Size([200, 2])
    >>> y.shape
    torch.Size([200])
    >>> y[:5]
    tensor([0., 0., 0., 0., 0.])
    >>> y[-5:]
    tensor([1., 1., 1., 1., 1.])
    """
    # Cluster 0 (y = 0)
    X0 = torch.randn(size, 2, dtype=dtype, generator=generator) * std0 + torch.tensor(mean0, dtype=dtype)
    y0 = torch.zeros(size, dtype=dtype)  # Labels for cluster 0

    # Cluster 1 (y = 1)
    X1 = torch.randn(size, 2, dtype=dtype, generator=generator) * std1 + torch.tensor(mean1, dtype=dtype)
    y1 = torch.ones(size, dtype=dtype)  # Labels for cluster 1

    X = torch.cat([X0, X1], dim=0)
    y = torch.cat([y0, y1], dim=0)

    return X, y