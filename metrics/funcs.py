import numpy as np


def topN_accuracy(y_true: np.array,
                  y_pred_onehot: np.array,
                  N: int):
    """ Top N accuracy """
    assert len(y_true.shape) == 1
    assert len(y_pred_onehot.shape) == 2
    assert y_true.shape[0] == y_pred_onehot.shape[0]
    assert y_pred_onehot.shape[1] >= N

    true_positive = 0
    for label, top_n_pred in zip(y_true, np.argsort(-y_pred_onehot, axis=-1)[:, :N]):
        if label in top_n_pred:
            true_positive += 1

    accuracy = true_positive / len(y_true)

    return accuracy
