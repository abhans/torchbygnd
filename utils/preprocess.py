import torch

# Function Definitions
def onehot(y: torch.Tensor, num_classes: int) -> torch.Tensor:
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
    means: list = [(-3, -3), (3, 3)],
    stds: list = [1.0, 1.0],
    labels: list = None,
    dtype=torch.float32,
    generator=None,
):
    """
    Generate multiple distinct clusters of data for classification tasks.

    Parameters
    ----------
    size : int
        Number of points per cluster.
    means : list of tuples, optional
        A list of tuples representing the means (centers) for each cluster.
        Default is ``[(-3, -3), (3, 3)]`` for two clusters.
    stds : list of floats, optional
        A list of standard deviations for each cluster. Must match the length of ``means``.
        Default is ``[1.0, 1.0]``.
    labels : list, optional
        A list of labels for each cluster. Must match the length of ``means``.
        If None, defaults to cluster indices (0, 1, 2, ...). Default is ``None``.
    dtype : torch.dtype, optional
        Data type of the output tensors. Default is ``torch.float32``.
    generator : torch.Generator, optional
        Random generator for reproducibility. Default is ``None``.

    Returns
    -------
    X : torch.Tensor
        Feature matrix of shape ``(len(means) * size, 2)`` containing the generated data points.
    y : torch.Tensor
        Labels of shape ``(len(means) * size,)`` corresponding to the cluster indices.

    Examples
    --------
    >>> X, y = clusters(size=100, means=[(-3, -3), (3, 3)], stds=[0.5, 0.5], labels=[1, -1])
    >>> X.shape
    torch.Size([200, 2])
    >>> y.shape
    torch.Size([200])
    >>> y[:5]
    tensor([1., 1., 1., 1., 1.])
    >>> y[-5:]
    tensor([-1., -1., -1., -1., -1.])
    """
    if len(means) != len(stds):
        raise ValueError("The number of means must match the number of standard deviations.")

    if labels is not None and len(means) != len(labels):
        raise ValueError("The number of means must match the number of labels.")

    Xs: list = []
    ys: list = []

    for idx, (mean, std) in enumerate(zip(means, stds)):
        X = torch.randn(size, 2, dtype=dtype, generator=generator) * std + torch.tensor(mean, dtype=dtype)
        
        # Assign labels from `labels`
        if labels is None:
            label = idx
        else:
            label = labels[idx]
        
        y = torch.full((size,), label, dtype=dtype)
        
        Xs.append(X)
        ys.append(y)

    X = torch.cat(Xs, dim=0)
    y = torch.cat(ys, dim=0)

    return X, y
