import torch
import torch.nn as nn


def sequence_mask(X, valid_len, value: int = 0):
    """
    Mask irrelevant entries in X to `value`
    :param X: shape (N, m, ...)
    :param valid_len: shape (N, ), X_i's m values if after valid_len_i, irrelevant
    """
    # m
    maxlen = X.size(1)
    # (1, m) < (N, 1)
    # broadcast (N, m)
    # LHS each row copy of 1 ... m
    # RHS each column copy of valid_len
    # True if less than valid_len's value, thus relevant
    mask = torch.arange(maxlen,
                        dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    print(mask)

    X[~mask] = value
    return X


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """The softmax cross-entropy loss with masks."""

    # `pred` shape: (`batch_size`, `num_steps`, `vocab_size`)
    # `label` shape: (`batch_size`, `num_steps`)
    # `valid_len` shape: (`batch_size`,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        print(pred.permute(0,2,1).shape, label.shape)
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss

