import torch
import torch.nn as nn
from typing import Optional


def sequence_mask(valid_len, max_len: Optional[int] = None):
    """
    find valid_len's corresponding mask
    Args:
        valid_len: (d_1, ..., d_n), each ele is valid len for some matrix of shape (d_1, ..., d_n, d),
            eg. if valid_len[1,2,3] = 6, then matrix[1,2,3] (an array) after 6th ele would marked irrelevant, thus False
        max_len: if None `max_len = max(valid_len)` the max of all valid_len ele

    Returns:
        mask: (d_1, ..., d_n, max_len),
            mask[i_1, ..., i_n, i] = i < valid_len[i_1, ..., i_n]
            thus matrix[~mask] would be those irrelevant fields
    """
    if not max_len:
        max_len = torch.max(valid_len)
    # (1, m) < (d_1, ..., d_n , 1)
    # broadcast (d_1, ..., d_n, m)
    # LHS each row (high-dim) copy of 1 ... m
    # RHS each column copy of valid_len
    # True if less than valid_len's value, thus relevant
    mask = torch.arange(max_len,
                        dtype=torch.float32,
                        device=valid_len.device)[None, :] < valid_len[..., None]

    return mask

def fill_sequence_mask(X, valid_len, value: int = 0):
    """
    fill mask of X to be `value`
    Args:
        X: input tensor (d_1, ..., d_n, d)
        valid_len: (d_1, ..., d_n), valid_len[i_1, ..., i_n] (int) is the valid len of X[i_1, ..., i_n] (row)
        value: scalar

    Returns:
        None, inplace change
    """
    mask = sequence_mask(valid_len, X.size(-1))
    X[~mask] = value

