import torch
import torch.nn as nn
from torcherist.tensor import *

def test_sequence_mask():
    # X = torch.tensor([[1, 2, 3], [4, 5, 6]])
    valid_len = torch.tensor([1,3,2])
    assert torch.all(torch.eq(
        sequence_mask(torch.tensor([1,3,2]), 5),
        torch.tensor(
            [[True, False, False, False, False],
             [True, True, True, False, False],
             [True, True, False, False, False]], dtype=torch.bool
        ))
    )

    assert torch.all(torch.eq(
        sequence_mask(torch.tensor([[1,3],[2,0]])),
        torch.tensor(
            [[[True, False, False],
              [True, True, True]],
             [[True, True, False],
              [False, False, False]]], dtype=torch.bool
        )
    ))

    # masked softmax cross entropy
    pred = torch.ones(3, 4, 10)
    label = torch.ones((3, 4), dtype=torch.long)
    valid_len = torch.tensor([4, 2, 0])
    weights = torch.ones_like(label)
    # fill irrelevant items in weights to 0
    fill_sequence_mask(weights, valid_len)

    unweighted_loss = nn.CrossEntropyLoss(reduction="none")(
                    pred.permute(0, 2, 1), label)
    weighted_loss = (unweighted_loss * weights).mean(dim=1)
    assert torch.allclose(weighted_loss, torch.tensor([2.3026, 1.1513, 0.000]))
