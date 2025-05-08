import numpy as np
from sklearn.metrics import mean_absolute_error


def nmae(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """
    Calculate the Normalized Mean Absolute Error (NMAE) between the true and predicted values.
    Normalize by the mean of the target values.
    """
    assert y_test.shape == y_pred.shape, "y_test and y_pred must have the same shape"
    return mean_absolute_error(y_test, y_pred) / (np.mean(np.abs(y_test)) + eps)
