import torch
from torcher.tensor import *

def test_sequence_mask():
    X = torch.tensor([[1, 2, 3], [4, 5, 6]])
    assert torch.allclose(sequence_mask(X, torch.tensor([1, 2])), torch.tensor([[1, 0, 0], [4, 5, 0]]))
    loss = MaskedSoftmaxCELoss()
    print(loss(torch.ones(3, 4, 10), torch.ones((3, 4), dtype=torch.long),
               torch.tensor([4, 2, 0])))
    X = torch.ones(2, 3, 4)
    XX = sequence_mask(X, torch.tensor([1, 2]), value=-1)
    # assert XX[0, ]
